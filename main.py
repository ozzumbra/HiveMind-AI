import torch
from honeycomb_conv import HoneycombConv2d

def main():
    model = HoneycombConv2d(in_channels=3, out_channels=16, radius=1)
    dummy_input = torch.randn(1, 3, 64, 64)
    
    output = model(dummy_input)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()

