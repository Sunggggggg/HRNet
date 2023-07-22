import argparse, torch, cv2, random
import matplotlib.pyplot as plt
from torchvision import transforms

from hrnet import HRNet

def arg_parse():
    desc = "HRNet Test with single image"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--weight', type=str, required=True,
                        help='Pre-trained Model Weights')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Test image directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')

def img2tensor(img, image_size = 224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    convert_tensor = transforms.ToTensor()
    test_input = convert_tensor(img).unsqueeze(0)

    return test_input, img

def extract_featuremaps(model, image):
    x = model.init_conv(x)
    x = model.first_layer(x)
    x_list = [m(x) for m in model.first_transition]
    for i in range(model.num_stage - 1):
        x_list = model.hr_blocks[i](x_list)

    res_list = [x_list[0]]
    for t, m in zip(x_list[1:], model.up_samplings):
        res_list.append(m(t))
    return res_list

def show_featuremaps(res_list, num = 5) :
    '''
    res_list : four stage feature map
        res_list[0].shape = (B, midC, image_size, image_size)
        res_list[1].shape = (B, midC * 2, image_size, image_size)
        res_list[2].shape = (B, midC * 4, image_size, image_size)
        res_list[3].shape = (B, midC * 8, image_size, image_size)
    '''
    N = len(res_list)
    fig, axs = plt.subplots(N, num)
    for stage, res in enumerate(res_list) :
        featuremap = res[0, :num, :, :]
        for idx, k in enumerate(featuremap) :
            axs[stage, idx].imshow(featuremap.detach().cpu().numpy())
            axs[stage, idx].axis('off')
    plt.savefig(output_dir)
    plt.close()

if __name__ == "__main__":
    arg = arg_parse()

    model_weight = arg.weight
    image_dir = arg.image_dir
    output_dir = arg.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_weight).to(device)
    image = cv2.imread(image_dir)
    x, ori = img2tensor(image)

    res_list = extract_featuremaps(model = model, image = x)
    
    show_featuremaps(res_list, output_dir)

    
    

    

