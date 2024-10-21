import argparse
from craft import CRAFT
from craft_utils import copyStateDict
import torch.onnx


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cpu")
    net = CRAFT()   

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=device, weights_only=True)))
    net = net.to(device)
    net.eval()

    dummy_input = torch.randn(1, 3, 1280, 1280).to(device)

    onnx_output_path = "craft_model.onnx"
    torch.onnx.export(
        net,  
        dummy_input,
        onnx_output_path,  
        export_params=True, 
        opset_version=11,  
        do_constant_folding=True,  
        input_names=['input'],  
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
    )

    print(f"Model has been successfully exported to {onnx_output_path}")
