import sys
sys.path.append('ldm/ldm/')
sys.path.append('moler_reference')
sys.path.append('ldm')
from ldm.evaluation_utils import MoLeRGenerator, LDMGenerator

if __name__ == '__main__':
    number_samples = 10000
    generator = MoLeRGenerator(
        ckpt_file_path='/data/conghao001/FYP/DrugDiscovery/first_stage_models/2023-03-03_09_30_01.589479/epoch=12-val_loss=0.46.ckpt',
        layer_type='FiLMConv',
        model_type='vae',
        using_lincs=False,
        using_gp=False,
        using_wasserstein_loss=False,
        device='cuda:0',
    )

    samples = generator.generate(number_samples)

    with open('generation_res/moler_samples.txt', 'w') as f:
        for sample in samples:
            f.write(sample + '\n')