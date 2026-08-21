import torch

from models.cnn import TFBS_CNN

from config import DEVICE


def main():

    print("================================")
    print("CNN SHAPE TEST")
    print("================================")

    # Create model
    model = TFBS_CNN()

    model = model.to(DEVICE)

    # Simulate one batch
    #
    # 64 sequences
    # 4 one-hot channels
    # 101 DNA bases

    x = torch.randn(
        64,
        4,
        101
    ).to(DEVICE)

    print("\nInput shape:")
    print(x.shape)

    # Forward pass
    output = model(x)

    print("\nOutput shape:")
    print(output.shape)

    # Check expected shape
    assert output.shape == (
        64,
        1
    )

    print("\n✓ CNN forward pass successful")

    print("✓ Input shape:  [64, 4, 101]")

    print("✓ Output shape: [64, 1]")

    print("\nCNN architecture is working.")


if __name__ == "__main__":

    main()