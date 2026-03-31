import os
import numpy as np
import onnxruntime as ort

class MindSporeVAE:
    def __init__(self, onnx_path, device='NPU'):
        self.onnx_path = onnx_path
        
        if device == 'NPU' and 'NPUExecutionProvider' in ort.get_available_providers():
            providers = ['NPUExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"VAE loaded from {onnx_path}, using providers: {providers}")
    
    def decode(self, latent):
        if hasattr(latent, 'asnumpy'):
            latent_np = latent.asnumpy()
        else:
            latent_np = np.array(latent)
        
        latent_np = latent_np.astype(np.float32)
        
        output = self.session.run(None, {'latent': latent_np})[0]
        
        return output


def load_onnx_model(onnx_path):
    """加载ONNX模型并验证"""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print(f"Model: {onnx_path}")
    print(f"Inputs: {[(inp.name, inp.shape, inp.type) for inp in inputs]}")
    print(f"Outputs: {[(out.name, out.shape, out.type) for out in outputs]}")
    
    return session


if __name__ == "__main__":
    vae = MindSporeVAE("mindspore/vae_decoder.onnx")
    
    latent = np.random.randn(1, 4, 32, 32).astype(np.float32)
    decoded = vae.decode(latent)
    print(f"Decoded shape: {decoded.shape}")
