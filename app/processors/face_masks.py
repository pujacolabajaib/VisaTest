from typing import TYPE_CHECKING, Tuple

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import kornia.morphology as morph
from collections import defaultdict
import math # Ensure math is imported

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

class FaceMasks:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def apply_occlusion(self, img, amount):
        # img is expected to be CHW, 0-255 range, uint8 from frame
        img_normalized = img.float() / 255.0 # Normalize 0-1
        if img_normalized.dim() == 3: # Ensure batch dim
            img_normalized = torch.unsqueeze(img_normalized, 0) # 1CHW
        
        outpred = torch.ones((1, 1, 256, 256), dtype=torch.float32, device=self.models_processor.device).contiguous() # Ensure 1,1,H,W for model output

        # Assuming run_occluder expects 1,C,H,W normalized input and fills output of shape 1,1,H,W
        self.models_processor.run_occluder(img_normalized, outpred) # img_normalized is 1,C,256,256

        # Output of run_occluder is in outpred, shape [1,1,256,256]
        # For dilation/erosion, it's easier to work with masks where 1 is area to keep
        # Assuming occluder output: 1 for occluded (remove), 0 for not occluded (keep)
        # So, we might need to invert based on how occluder model behaves.
        # If outpred > 0 means "occluded", then (outpred <= 0) means "keep".
        # Let's assume outpred from model means: 1 = occluded, 0 = not occluded.
        # We want a mask of areas to KEEP. So, mask = (outpred == 0)
        
        processed_mask = (outpred < 0.5).float() # Convert to binary mask: 1 for non-occluded (keep), 0 for occluded (remove)

        if amount != 0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device) # Kernel for 2D conv
            # Ensure processed_mask is 4D for conv2d: [N, C, H, W] where C=1
            if processed_mask.dim() == 2: # HW
                processed_mask = processed_mask.unsqueeze(0).unsqueeze(0) # 1,1,H,W
            elif processed_mask.dim() == 3 and processed_mask.shape[0]==1: # 1,H,W (assuming single channel mask)
                processed_mask = processed_mask.unsqueeze(1) # 1,1,H,W
            
            if amount > 0: # Dilate the "keep" area
                for _ in range(int(amount)):
                    processed_mask = torch.nn.functional.conv2d(processed_mask, kernel, padding=(1, 1))
                    processed_mask = torch.clamp(processed_mask, 0, 1)
            elif amount < 0: # Erode the "keep" area
                # To erode "keep" area, invert, dilate, then invert back
                processed_mask_inverted = 1.0 - processed_mask
                for _ in range(int(-amount)): # Iterate positive number of times
                    processed_mask_inverted = torch.nn.functional.conv2d(processed_mask_inverted, kernel, padding=(1, 1))
                    processed_mask_inverted = torch.clamp(processed_mask_inverted, 0, 1)
                processed_mask = 1.0 - processed_mask_inverted
        
        # Return as 1,H,W for consistency with other masks
        if processed_mask.dim() == 4 and processed_mask.shape[0] == 1 and processed_mask.shape[1] == 1:
             processed_mask = processed_mask.squeeze(1) # from [1,1,H,W] to [1,H,W]
        elif processed_mask.dim() == 2: # HW
             processed_mask = processed_mask.unsqueeze(0) # to [1,H,W]

        return processed_mask


    def run_occluder(self, image, output):
        # image: 1,C,H,W normalized
        # output: 1,1,H,W pre-allocated
        if not self.models_processor.models['Occluder']:
            self.models_processor.models['Occluder'] = self.models_processor.load_model('Occluder')

        io_binding = self.models_processor.models['Occluder'].io_binding()
        io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.shape, buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.shape, buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            if hasattr(self.models_processor, 'syncvec') and callable(self.models_processor.syncvec.cpu): # Defensive check
                self.models_processor.syncvec.cpu() 
        self.models_processor.models['Occluder'].run_with_iobinding(io_binding)

    def _process_xseg_variant(self, base_mask_clone, amount, blur_slider_value, device):
        processed_mask = base_mask_clone # base_mask_clone is already [1, 1, H, W]
        
        if amount != 0: 
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device) # Kernel for 2D conv
            if amount > 0: # Dilate "keep" area
                for _ in range(int(amount)):
                    processed_mask = torch.nn.functional.conv2d(processed_mask, kernel, padding=(1, 1))
                    processed_mask = torch.clamp(processed_mask, 0, 1)
            elif amount < 0: # Erode "keep" area
                processed_mask_inverted = 1.0 - processed_mask 
                for _ in range(int(-amount)): 
                    processed_mask_inverted = torch.nn.functional.conv2d(processed_mask_inverted, kernel, padding=(1, 1))
                    processed_mask_inverted = torch.clamp(processed_mask_inverted, 0, 1)
                processed_mask = 1.0 - processed_mask_inverted
        
        if blur_slider_value > 0:
            kernel_size = blur_slider_value * 2 + 1
            if kernel_size % 2 == 0: kernel_size +=1 # Ensure odd
            sigma = (blur_slider_value + 1) * 0.2
            if sigma <= 0: sigma = 0.1 
            gauss = transforms.GaussianBlur(kernel_size, sigma=sigma)
            processed_mask = gauss(processed_mask)
        
        return processed_mask # Shape [1,1,H,W]

    def apply_dfl_xseg(self, original_input_img_256, original_input_img_512, kps_5_for_masking, mouth_mask_from_parser_512, parameters):
        device = self.models_processor.device
        t256_resize = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # --- Generate face_mask_full_256 ---
        face_mask_full_512 = None
        if parameters.get("FaceParserEnableToggle", False):
            # Assuming apply_face_parser returns: out_parse, face_mask_texture, bg_mask_texture, combined_mouth_mask
            # We need bg_mask from the "original" mode.
            _, _, bg_mask_orig_512, _ = self.apply_face_parser(original_input_img_512, parameters, mode="original")
            if isinstance(bg_mask_orig_512, torch.Tensor) and bg_mask_orig_512.numel() > 0:
                face_mask_full_512 = 1.0 - bg_mask_orig_512.to(device) # Invert background to get foreground
            else: 
                print("Warning: FaceParser enabled but bg_mask_orig_512 was not valid. Falling back to kps mask for DFLXSeg.")

        if face_mask_full_512 is None: # If FaceParser is off or bg_mask was invalid
            min_x = int(kps_5_for_masking[:, 0].min())
            max_x = int(kps_5_for_masking[:, 0].max())
            min_y = int(kps_5_for_masking[:, 1].min())
            max_y = int(kps_5_for_masking[:, 1].max())
            
            face_width = max_x - min_x; face_height = max_y - min_y
            padding_x = int(face_width * 0.25); padding_y = int(face_height * 0.30)

            rect_min_x = max(0, min_x - padding_x); rect_max_x = min(511, max_x + padding_x) # Assuming 512x512 input for kps
            rect_min_y = max(0, min_y - padding_y); rect_max_y = min(511, max_y + padding_y)

            face_mask_full_512 = torch.zeros((1, 512, 512), dtype=torch.float32, device=device)
            face_mask_full_512[:, rect_min_y:rect_max_y+1, rect_min_x:rect_max_x+1] = 1.0
            
            blur_kernel_s = 21 ; blur_sigma_s = 7.0 # Soften the rect mask
            if blur_kernel_s > 0:
                 if blur_kernel_s % 2 == 0: blur_kernel_s +=1
                 gaussian_blur_rect = transforms.GaussianBlur(kernel_size=blur_kernel_s, sigma=blur_sigma_s)
                 face_mask_full_512 = gaussian_blur_rect(face_mask_full_512)
        
        face_mask_full_256 = t256_resize(face_mask_full_512.to(device))

        # --- Prepare mouth_mask_256 ---
        if isinstance(mouth_mask_from_parser_512, torch.Tensor) and mouth_mask_from_parser_512.numel() > 0:
            current_mouth_mask_256 = t256_resize(mouth_mask_from_parser_512.to(device))
        else:
            current_mouth_mask_256 = torch.zeros((1, 256, 256), dtype=torch.float32, device=device)

        # --- Retrieve Parameters ---
        amount_inside_face = -parameters.get("DFLXSegSizeInsideFaceSlider", 0)
        blur_inside_face_param = parameters.get("DFLXSegBlurInsideFaceSlider", 0)
        amount_outside_face = -parameters.get("DFLXSegSizeOutsideFaceSlider", 0)
        blur_outside_face_param = parameters.get("DFLXSegBlurOutsideFaceSlider", 0)
        amount_mouth = -parameters.get("DFLXSeg2SizeSlider", 0)
        blur_mouth_param = parameters.get("XSeg2BlurSlider", 0)

        # --- Base XSeg Mask ---
        img_for_xseg = original_input_img_256.type(torch.float32) # Should be CHW, 0-255
        img_for_xseg_norm = torch.div(img_for_xseg, 255.0) # Normalize 0-1
        if img_for_xseg_norm.dim() == 3:
            img_for_xseg_norm = torch.unsqueeze(img_for_xseg_norm, 0) # Add batch dim if needed: 1,C,H,W
        
        base_outpred_raw = torch.ones((1,1,256,256), dtype=torch.float32, device=device).contiguous() # Ensure shape for run_dfl_xseg output
        self.run_dfl_xseg(img_for_xseg_norm, base_outpred_raw) # Expects normalized img_for_xseg

        base_outpred = torch.clamp(base_outpred_raw, min=0.0, max=1.0)
        base_outpred[base_outpred < 0.1] = 0.0 # Ensure complete removal for low values
        base_outpred = 1.0 - base_outpred  # Inverted: mask of areas to KEEP. Shape [1,1,256,256]
        
        # --- Process Inside and Outside Masks ---
        outpred_inside = self._process_xseg_variant(base_outpred.clone(), amount_inside_face, blur_inside_face_param, device)
        outpred_outside = self._process_xseg_variant(base_outpred.clone(), amount_outside_face, blur_outside_face_param, device)

        # --- Combine Using Full Face Mask ---
        # face_mask_full_256 is [1,256,256], outpred_inside/outside are [1,1,256,256] or [1,256,256]
        # Ensure consistent channel dimensions for torch.where
        if face_mask_full_256.dim() == 3 and face_mask_full_256.shape[0] == 1: # If [1, H, W]
             face_mask_full_256_expanded = face_mask_full_256.unsqueeze(0) # to [1,1,H,W] if not already
        elif face_mask_full_256.dim() == 2: # H, W
             face_mask_full_256_expanded = face_mask_full_256.unsqueeze(0).unsqueeze(0) # to [1,1,H,W]
        else: # Assuming [1,1,H,W] or compatible
             face_mask_full_256_expanded = face_mask_full_256
        
        if outpred_inside.dim() == 3: outpred_inside = outpred_inside.unsqueeze(0)
        if outpred_outside.dim() == 3: outpred_outside = outpred_outside.unsqueeze(0)

        combined_pred = torch.where(face_mask_full_256_expanded > 0.5, outpred_inside, outpred_outside)

        # --- Process Mouth Area (conditionally) ---
        if parameters.get("XSegMouthEnableToggle", False) and amount_mouth != amount_inside_face :
            outpred_mouth_area = self._process_xseg_variant(base_outpred.clone(), amount_mouth, blur_mouth_param, device)
            if outpred_mouth_area.dim() == 3: outpred_mouth_area = outpred_mouth_area.unsqueeze(0) # ensure [1,1,H,W]
            
            if current_mouth_mask_256.dim() == 3 and current_mouth_mask_256.shape[0] == 1: # [1,H,W]
                current_mouth_mask_256_expanded = current_mouth_mask_256.unsqueeze(0)
            elif current_mouth_mask_256.dim() == 2: # H,W
                current_mouth_mask_256_expanded = current_mouth_mask_256.unsqueeze(0).unsqueeze(0)
            else: # Assuming [1,1,H,W] or compatible
                current_mouth_mask_256_expanded = current_mouth_mask_256

            combined_pred = torch.where(current_mouth_mask_256_expanded > 0.9, outpred_mouth_area, combined_pred)
        
        final_outpred = torch.reshape(combined_pred, (1, 256, 256)) # Final shape [1, H, W]
        return final_outpred


    def run_dfl_xseg(self, image, output):
        if not self.models_processor.models['XSeg']:
            self.models_processor.models['XSeg'] = self.models_processor.load_model('XSeg')

        io_binding = self.models_processor.models['XSeg'].io_binding()
        # Ensure image shape is (1, 3, H, W) for the model
        if image.dim() == 3: # If C, H, W
            image = image.unsqueeze(0) # Add batch dim: 1, C, H, W
        
        io_binding.bind_input(name='in_face:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.shape, buffer_ptr=image.data_ptr())
        # Ensure output shape is (1, 1, H, W) for the model
        # The 'output' tensor passed to this function is already pre-shaped to (1,1,256,256) typically
        io_binding.bind_output(name='out_mask:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.shape, buffer_ptr=output.data_ptr())


        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            if hasattr(self.models_processor, 'syncvec') and callable(self.models_processor.syncvec.cpu): # Defensive check
                self.models_processor.syncvec.cpu()
        self.models_processor.models['XSeg'].run_with_iobinding(io_binding)
        
    def apply_face_parser(self, img, parameters, mode): # img is CHW, 0-255
        FaceAmount = -parameters.get("BackgroundParserSlider",0)
        FaceAmountTexture = -parameters.get("BackgroundParserTextureSlider",0)

        img_normalized = img.float() / 255.0 # Normalize 0-1
        img_normalized = v2.functional.normalize(img_normalized, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if img_normalized.dim() == 3: # C,H,W
             img_normalized = img_normalized.unsqueeze(0) # Add batch: 1,C,H,W
        # Ensure img is 512x512 for face parser model
        if img_normalized.shape[2] != 512 or img_normalized.shape[3] != 512:
            t_resize_512 = v2.Resize((512,512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
            img_for_parser = t_resize_512(img_normalized)
        else:
            img_for_parser = img_normalized


        outpred_parser = torch.empty((1, 19, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
        self.run_faceparser(img_for_parser, outpred_parser)
        outpred_parser_argmax = torch.argmax(outpred_parser.squeeze(0), 0) # HW

        def create_mask_fp(attributes, iterations, current_outpred_argmax):
            mask = torch.isin(current_outpred_argmax, torch.tensor(attributes, device=current_outpred_argmax.device)).float() # HW
            if iterations == 0 : # No processing if iteration is 0
                return mask.unsqueeze(0) # Return as 1HW
            
            mask_processed = mask.unsqueeze(0).unsqueeze(0) # 1,1,H,W for morph ops
            
            # Use absolute iterations for loop, invert mask before/after for negative iterations
            invert_mask = iterations < 0
            abs_iterations = abs(iterations)

            if invert_mask:
                mask_processed = 1.0 - mask_processed
            
            kernel_morph = torch.ones((3, 3), device=mask_processed.device)
            for _ in range(abs_iterations):
                mask_processed = morph.dilation(mask_processed, kernel=kernel_morph) # kornia dilation
            
            if invert_mask: # Invert back if iterations were negative
                mask_processed = 1.0 - mask_processed
                
            return torch.clamp(mask_processed.squeeze(0), 0, 1) # Return 1HW, clamped


        face_attrs_map = {
            1: parameters.get('FaceParserSlider',0), 2: parameters.get('LeftEyebrowParserSlider',0),
            3: parameters.get('RightEyebrowParserSlider',0), 4: parameters.get('LeftEyeParserSlider',0),
            5: parameters.get('RightEyeParserSlider',0), 6: parameters.get('EyeGlassesParserSlider',0),
            10: parameters.get('NoseParserSlider',0), 11: parameters.get('MouthParserSlider',0), # Mouth region (inner)
            12: parameters.get('UpperLipParserSlider',0), 13: parameters.get('LowerLipParserSlider',0),
            14: parameters.get('NeckParserSlider',0), 17: parameters.get('HairParserSlider',0),
        }
        face_attrs_tex_map = {
            2: parameters.get('EyebrowParserTextureSlider',0), 3: parameters.get('EyebrowParserTextureSlider',0),
            4: parameters.get('EyeParserTextureSlider',0), 5: parameters.get('EyeParserTextureSlider',0),
            10: parameters.get('NoseParserTextureSlider',0), 11: parameters.get('MouthParserTextureSlider',0),
            12: parameters.get('MouthParserTextureSlider',0), 13: parameters.get('MouthParserTextureSlider',0),
            14: parameters.get('NeckParserTextureSlider',0),
        }
        mouth_attrs_map = {
            11: parameters.get('XsegMouthParserSlider',0), 12: parameters.get('XsegUpperLipParserSlider',0),
            13: parameters.get('XsegLowerLipParserSlider',0)
        }
        bg_attrs_list = [0, 15, 16, 18] # Standard background attributes, (14-Neck, 17-Hair sometimes included in FG)

        def group_and_combine_fp(attr_map, current_outpred_argmax_local):
            # ... (group_and_combine logic from previous version, ensuring it uses current_outpred_argmax_local) ...
            # This helper should sum masks created by create_mask_fp
            # Example:
            result = torch.zeros((1, 512, 512), dtype=torch.float32, device=current_outpred_argmax_local.device)
            grouped = defaultdict(list)
            for attr, dil in attr_map.items():
                if dil != 0: # Only process if dilation/erosion value is non-zero
                    grouped[dil].append(attr)
            for dil, attrs_list in grouped.items():
                # create_mask_fp returns 1,H,W
                result += create_mask_fp(attrs_list, dil, current_outpred_argmax_local)
            return torch.clamp(result, 0, 1)


        # For 'out_parse' (general face mask for XSeg combination)
        out_parse_sum = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        if parameters.get("FaceParserEnableToggle", False):
            out_parse_sum = group_and_combine_fp(face_attrs_map, outpred_parser_argmax)
            blur_val = parameters.get('FaceBlurParserSlider',0); 
            if blur_val > 0: 
                k = blur_val*2+1; 
                if k % 2 == 0: k+=1
                sigma=(blur_val+1)*0.2; 
                out_parse_sum = transforms.GaussianBlur(k, max(sigma,0.1))(out_parse_sum)
        
        # For 'combined_mouth_mask'
        combined_mouth_mask_512 = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        if parameters.get("XSegMouthEnableToggle", False) or parameters.get("FaceEditorEnableToggle", False): # if mouth parts needed
            combined_mouth_mask_512 = group_and_combine_fp(mouth_attrs_map, outpred_parser_argmax)
            # No blur usually applied to mouth mask for XSeg, but can be added if needed

        # For 'bg_parse' (background mask)
        bg_parse_sum = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        if parameters.get("FaceParserEnableToggle", False) and FaceAmount != 0 :
            bg_parse_sum = create_mask_fp(bg_attrs_list, FaceAmount, outpred_parser_argmax)
            blur_val = parameters.get('BackgroundBlurParserSlider',0)
            if blur_val > 0: 
                k = blur_val*2+1; 
                if k % 2 == 0: k+=1
                sigma=(blur_val+1)*0.2; 
                bg_parse_sum = transforms.GaussianBlur(k, max(sigma,0.1))(bg_parse_sum)

        # This is the main mask used by swap_core if FaceParserEnableToggle is on
        # It's what used to be (1 - torch.clamp(out_parse + bg_parse, 0, 1))
        # Now, it should be a mask of the foreground.
        final_out_parse_mask = torch.clamp(out_parse_sum - bg_parse_sum, 0, 1) if parameters.get("FaceParserEnableToggle", False) else torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        
        # For 'face_mask_texture'
        face_mask_texture_512 = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        use_texture_face_mask = any(parameters.get(s,0)!=0 for s in ['EyebrowParserTextureSlider', 'EyeParserTextureSlider', 'NoseParserTextureSlider', 'MouthParserTextureSlider', 'NeckParserTextureSlider'])
        if (parameters.get("TransferTextureEnableToggle",False) or parameters.get("DifferencingEnableToggle",False)) and parameters.get("ExcludeMaskEnableToggle",False) and use_texture_face_mask:
            face_mask_texture_512 = group_and_combine_fp(face_attrs_tex_map, outpred_parser_argmax)
            blur_val = parameters.get('FaceParserBlurTextureSlider',0)
            if blur_val > 0: 
                k=blur_val*2+1; 
                if k % 2 == 0: k+=1
                sigma=(blur_val+1)*0.2; 
                face_mask_texture_512 = transforms.GaussianBlur(k,max(sigma,0.1))(face_mask_texture_512)

        # For 'bg_mask_texture'
        bg_mask_texture_512 = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        if (parameters.get("TransferTextureEnableToggle",False) or parameters.get("DifferencingEnableToggle",False)) and parameters.get("ExcludeMaskEnableToggle",False) and mode == "original" and FaceAmountTexture != 0:
            bg_mask_texture_512 = create_mask_fp(bg_attrs_list, FaceAmountTexture, outpred_parser_argmax) # bg_attributes_texture was same as bg_attributes
            blur_val = parameters.get('FaceParserBlurBGTextureSlider',0)
            if blur_val > 0: 
                k=blur_val*2+1; 
                if k % 2 == 0: k+=1
                sigma=(blur_val+1)*0.2; 
                bg_mask_texture_512 = transforms.GaussianBlur(k,max(sigma,0.1))(bg_mask_texture_512)
        
        return final_out_parse_mask, face_mask_texture_512, bg_mask_texture_512, combined_mouth_mask_512


    def run_faceparser(self, image, output): # image is 1,C,H,W normalized ; output is 1,19,H,W
        # ... (original content of run_faceparser - unchanged) ...
        if not self.models_processor.models['FaceParser']:
            self.models_processor.models['FaceParser'] = self.models_processor.load_model('FaceParser')

        image = image.contiguous()
        io_binding = self.models_processor.models['FaceParser'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.shape, buffer_ptr=image.data_ptr()) # Use image.shape
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=output.shape, buffer_ptr=output.data_ptr()) # Use output.shape

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            if hasattr(self.models_processor, 'syncvec') and callable(self.models_processor.syncvec.cpu): # Defensive check
                self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceParser'].run_with_iobinding(io_binding)


    def run_CLIPs(self, img, CLIPText, CLIPAmount): # img is CHW 0-255
        # ... (original content of run_CLIPs - unchanged) ...
        device = img.device
        if not self.models_processor.clip_session:
            self.models_processor.clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.models_processor.clip_session.eval()
            self.models_processor.clip_session.load_state_dict(torch.load(f'{models_dir}/rd64-uni-refined.pth', weights_only=True, map_location=device), strict=False) # Added map_location
            self.models_processor.clip_session.to(device)

        clip_mask_out = torch.ones((1, 352, 352), device=device) # Ensure it's (1,H,W) for consistency
        img_float_norm = img.float() / 255.0
        
        transform_clip = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352), antialias=True) # Added antialias
        ])
        CLIPimg_transformed = transform_clip(img_float_norm) # Input CHW
        if CLIPimg_transformed.dim() == 3:
             CLIPimg_transformed = CLIPimg_transformed.unsqueeze(0) # Ensure 1CHW for model

        if CLIPText != "":
            prompts = [p.strip() for p in CLIPText.split(',') if p.strip()] # Ensure prompts are clean
            if not prompts: # If all prompts were empty or just commas
                 return clip_mask_out 

            with torch.no_grad():
                preds = self.models_processor.clip_session(CLIPimg_transformed.repeat(len(prompts), 1, 1, 1), prompts)[0]
            
            current_clip_mask = 1.0 - torch.sigmoid(preds[0][0]) # HW
            for i in range(1, len(prompts)): # Start from 1 if len > 1
                current_clip_mask *= (1.0 - torch.sigmoid(preds[i][0]))
            
            thresh = CLIPAmount / 100.0
            clip_mask_out = (current_clip_mask > thresh).float().unsqueeze(0) # Add channel dim: 1HW
        return clip_mask_out


    def soft_oval_mask(self, height, width, center: Tuple[float, float], radius_x: float, radius_y: float, feather_radius: float =None):
        # ... (original content of soft_oval_mask - unchanged) ...
        if feather_radius is None or feather_radius <= 0: # Ensure feather_radius is positive
            feather_radius = max(1.0, max(radius_x, radius_y) / 2.0)

        y, x = torch.meshgrid(torch.arange(height, device=self.models_processor.device, dtype=torch.float32), 
                              torch.arange(width, device=self.models_processor.device, dtype=torch.float32), 
                              indexing='ij')
        
        # Ensure radii are not zero to avoid division by zero
        safe_radius_x = max(radius_x, 1.0)
        safe_radius_y = max(radius_y, 1.0)
        feather_radius = max(feather_radius, 1.0) # Ensure feather_radius is not zero for division

        normalized_distance = torch.sqrt(((x - center[0]) / safe_radius_x) ** 2 + ((y - center[1]) / safe_radius_y) ** 2)
        mask = torch.clamp((1.0 - normalized_distance) * (safe_radius_x / feather_radius), 0, 1) # Use 1.0 for float division
        return mask


    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        # ... (original content of restore_mouth - unchanged) ...
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])
        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)
        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)
        mouth_center[0] += x_offset
        mouth_center[1] += y_offset
        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y) # Assuming img_orig is CHW
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)
        if ymax <= ymin or xmax <= xmin: return img_swap # If region is invalid
        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        # Ensure radii are positive for soft_oval_mask
        oval_radius_x = max(1, xmax - xmin) // 2
        oval_radius_y = max(1, ymax - ymin) // 2
        # Center for soft_oval_mask is relative to the crop
        oval_center_x = (xmax - xmin) // 2 # Integer for center index
        oval_center_y = (ymax - ymin) // 2 # Integer for center index

        mouth_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (float(oval_center_x), float(oval_center_y)), # Center as float
                                            float(oval_radius_x), float(oval_radius_y), # Radii as float
                                            float(feather_radius)).to(img_orig.device)
        img_swap_mouth = img_swap[:, ymin:ymax, xmin:xmax] # Target region in img_swap
        # Ensure mouth_region_orig and img_swap_mouth are same size for blending
        if mouth_region_orig.shape != img_swap_mouth.shape: return img_swap # Skip if shapes mismatch
        blended_mouth = blend_alpha * img_swap_mouth + (1.0 - blend_alpha) * mouth_region_orig
        img_swap[:, ymin:ymax, xmin:xmax] = mouth_mask * blended_mouth + (1.0 - mouth_mask) * img_swap_mouth
        return img_swap


    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        # ... (original content of restore_eyes - unchanged, but check oval_mask calls like in restore_mouth) ...
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])
        left_eye[0] += x_offset; right_eye[0] += x_offset
        left_eye[1] += y_offset; right_eye[1] += y_offset
        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / max(size_factor, 0.1)) # Avoid div by zero
        radius_x_eye = int(base_eye_radius * radius_factor_x)
        radius_y_eye = int(base_eye_radius * radius_factor_y)
        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(eye_center_coords, r_x, r_y, orig_img_tensor, swap_img_tensor, blend_val, feather_val):
            ymin_e = max(0, eye_center_coords[1] - r_y)
            ymax_e = min(orig_img_tensor.size(1), eye_center_coords[1] + r_y)
            xmin_e = max(0, eye_center_coords[0] - r_x)
            xmax_e = min(orig_img_tensor.size(2), eye_center_coords[0] + r_x)
            if ymax_e <= ymin_e or xmax_e <= xmin_e: return swap_img_tensor
            eye_region_orig_tensor = orig_img_tensor[:, ymin_e:ymax_e, xmin_e:xmax_e]
            
            oval_r_x_e = max(1, xmax_e - xmin_e) // 2
            oval_r_y_e = max(1, ymax_e - ymin_e) // 2
            oval_c_x_e = (xmax_e - xmin_e) // 2 # Integer for center index
            oval_c_y_e = (ymax_e - ymin_e) // 2 # Integer for center index

            eye_mask_tensor = self.soft_oval_mask(ymax_e - ymin_e, xmax_e - xmin_e,
                                            (float(oval_c_x_e), float(oval_c_y_e)), # Center as float
                                            float(oval_r_x_e), float(oval_r_y_e), # Radii as float
                                            float(feather_val)).to(orig_img_tensor.device)
            img_swap_eye_region = swap_img_tensor[:, ymin_e:ymax_e, xmin_e:xmax_e]
            if eye_region_orig_tensor.shape != img_swap_eye_region.shape: return swap_img_tensor
            blended_eye_tensor = blend_val * img_swap_eye_region + (1.0 - blend_val) * eye_region_orig_tensor
            swap_img_tensor[:, ymin_e:ymax_e, xmin_e:xmax_e] = eye_mask_tensor * blended_eye_tensor + (1.0 - eye_mask_tensor) * img_swap_eye_region
            return swap_img_tensor

        img_swap = extract_and_blend_eye(left_eye, radius_x_eye, radius_y_eye, img_orig, img_swap, blend_alpha, feather_radius)
        img_swap = extract_and_blend_eye(right_eye, radius_x_eye, radius_y_eye, img_orig, img_swap, blend_alpha, feather_radius)
        return img_swap

    def apply_fake_diff(self, swapped_face, original_face, lower_thresh, lower_value, upper_thresh, upper_value, middle_value):
        # ... (original content of apply_fake_diff - unchanged) ...
        diff = torch.abs(swapped_face.float() - original_face.float()) # Ensure float for calculations
        sample = diff.reshape(-1)
        # Ensure sample size is not larger than numel, and handle empty tensor case
        num_elements = sample.numel()
        if num_elements == 0: return torch.zeros_like(swapped_face[0:1,:,:]).float() # Return single channel zero mask

        sample_size = min(50000, num_elements)
        if sample_size > 0 :
            sample_indices = torch.randint(0, num_elements, (sample_size,), device=diff.device)
            diff_max_q = torch.quantile(sample[sample_indices], 0.99)
        else: # Fallback if somehow num_elements was 0 leading to sample_size 0
            diff_max_q = diff.max() if num_elements > 0 else torch.tensor(0.0, device=diff.device)

        diff_clamped = torch.clamp(diff, max=diff_max_q)
        diff_min_val = diff_clamped.min()
        diff_max_val = diff_clamped.max()
        
        # Avoid division by zero if min and max are the same
        if diff_max_val - diff_min_val < 1e-6: # Effectively zero range
            diff_norm = torch.zeros_like(diff_clamped)
        else:
            diff_norm = (diff_clamped - diff_min_val) / (diff_max_val - diff_min_val)

        diff_mean_ch = diff_norm.mean(dim=0) # HW
        
        # Ensure thresholds are valid (lower < upper)
        actual_lower_thresh = min(lower_thresh, upper_thresh - 1e-6) # Ensure lower_thresh is less than upper_thresh
        actual_upper_thresh = max(upper_thresh, lower_thresh + 1e-6) # Ensure upper_thresh is greater than lower_thresh


        scale_low = diff_mean_ch / actual_lower_thresh if actual_lower_thresh > 1e-6 else torch.zeros_like(diff_mean_ch)
        result = torch.where(
            diff_mean_ch < actual_lower_thresh,
            lower_value + scale_low * (middle_value - lower_value),
            torch.empty_like(diff_mean_ch) # Placeholder, will be filled
        )
        
        # Denominator for middle_scale, ensure it's not zero
        denom_middle = actual_upper_thresh - actual_lower_thresh
        if denom_middle < 1e-6: # Effectively zero
            middle_scale = torch.zeros_like(diff_mean_ch)
        else:
            middle_scale = (diff_mean_ch - actual_lower_thresh) / denom_middle

        result = torch.where(
            (diff_mean_ch >= actual_lower_thresh) & (diff_mean_ch <= actual_upper_thresh),
            middle_value + middle_scale * (upper_value - middle_value),
            result
        )
        
        # Denominator for above_scale, ensure it's not zero
        denom_above = 1.0 - actual_upper_thresh
        if denom_above < 1e-6: # Effectively zero
            above_scale = torch.zeros_like(diff_mean_ch)
        else:
            above_scale = (diff_mean_ch - actual_upper_thresh) / denom_above
            
        result = torch.where(
            diff_mean_ch > actual_upper_thresh,
            upper_value + above_scale * (1.0 - upper_value), # Should be (1.0 - upper_value) if upper_value is max for this part
            result
        )
        return result.unsqueeze(0).clamp(0.0,1.0) # 1HW, clamped
```
