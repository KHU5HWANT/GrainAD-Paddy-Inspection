import torch
import numpy as np
from models.loss import SupConLoss, multiview_labels
from models.ICA import STNModule


def test_supcon_excludes_self_and_matches_manual_objective():
    x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    loss = SupConLoss(1)(x, torch.tensor([0, 0, 1, 1]))
    expected = torch.log(torch.exp(torch.tensor(1.)) + 2) - 1
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_no_positives_is_differentiable_zero():
    x = torch.randn(3, 7, requires_grad=True)
    loss = SupConLoss()(x, torch.arange(3))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(x.grad).all()


def test_labels_follow_kernel_not_view_or_modulo_five():
    # Non-five batch size catches the original modulo-5 error.
    augmented = torch.tensor([0, 1, 0] * 5)
    labels = multiview_labels(3, 5, augmented)
    assert labels[:15].tolist() == [0, 1, 2] * 5
    assert labels[15:30].tolist() == [3, 4, 5] * 5
    assert labels[30:].tolist() == [0, 4, 2] * 5


def test_stn_cpu_identity_and_gradient():
    stn = STNModule(3).eval()
    x = torch.randn(1, 3, 224, 224)
    y, theta = stn(x)
    torch.testing.assert_close(theta, torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]))
    torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)
    y.square().mean().backward()
    assert stn.fc[2].weight.grad is not None


def test_inference_uses_each_kernels_worst_view():
    from models.simplenet import SimpleNet, PatchMaker
    from torch import nn
    model = SimpleNet(torch.device('cpu'))
    model.forward_modules = nn.ModuleDict()
    model.pre_proj = 0
    model.add_ica = False
    model.discriminator = nn.Identity()
    model.patch_maker = PatchMaker(1, stride=1)
    model._embed = lambda x, **kw: (-x[:, :1, :1, :1].reshape(-1, 1), [[1, 1]])
    class Segmentor:
        def convert_to_segmentation(self, scores, features):
            return scores, features
    model.anomaly_segmentor = Segmentor()
    a = torch.zeros(2, 3, 1, 1); a[:, 0, 0, 0] = torch.tensor([9., 1.])
    b = torch.zeros(2, 3, 1, 1); b[:, 0, 0, 0] = torch.tensor([2., 8.])
    scores, *_ = model._predict([a, b])
    np.testing.assert_allclose(scores, [9., 8.])


def test_training_step_updates_adapter_and_keeps_backbone_frozen():
    from models.simplenet import SimpleNet
    from torch import nn
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer2 = nn.Conv2d(3, 4, 3, padding=1)
            self.layer3 = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        def forward(self, x):
            return self.layer3(self.layer2(x))
    model = SimpleNet(torch.device('cpu'))
    model.load(Backbone(), ['layer2', 'layer3'], torch.device('cpu'),
               (3, 8, 8), 8, 8, pre_proj=1, dsc_layers=2, dsc_hidden=8,
               meta_epochs=2, gan_epochs=1, visualization=False, visual_num=0,
               add_ica=False, five_split=True, add_scl=True, ablation=0)
    before = model.pre_projection.layers[0].weight.detach().clone()
    batch = {'image': [torch.randn(2, 3, 8, 8) for _ in range(5)],
             'aug_image': [torch.randn(2, 3, 8, 8) for _ in range(5)],
             'aug_mask': [torch.ones(2, 1, 8, 8) for _ in range(5)],
             'aug_labels': torch.ones(2, 5)}
    model._train_discriminator([batch])
    assert not torch.equal(before, model.pre_projection.layers[0].weight)
    assert all(p.grad is None for p in model.backbone.parameters())
    scores, masks, *_ = model._predict(batch['image'])
    assert len(scores) == 2 and np.asarray(masks).shape == (2, 8, 40)
    assert np.isfinite(scores).all()
    # Frozen modules stay out of optimization; checkpoint round trips preserve output.
    import io
    buffer = io.BytesIO()
    torch.save(model.discriminator.state_dict(), buffer)
    buffer.seek(0)
    model.discriminator.load_state_dict(torch.load(buffer, weights_only=True))
    restored, *_ = model._predict(batch['image'])
    np.testing.assert_allclose(scores, restored)


def test_dataset_small_split_and_five_view_augmentation(tmp_path):
    from PIL import Image
    from data.grain import GrainDataset
    folder = tmp_path / 'soybean' / 'train' / 'good'
    folder.mkdir(parents=True)
    Image.fromarray(np.full((32, 160, 3), 100, dtype=np.uint8)).save(folder / 'one.png')
    textures = tmp_path / 'dtd' / 'texture'
    textures.mkdir(parents=True)
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(textures / 'one.jpg')
    ds = GrainDataset(str(tmp_path), five_split=True, resize=[32, 32],
                      imagesize=[32, 32], padding=0, anomaly_source_path=str(tmp_path / 'dtd'))
    sample = ds[0]
    assert len(ds) == 1 and len(sample['image']) == 5
    assert sample['image'][0].shape == (3, 32, 32)
    assert set(torch.unique(sample['aug_mask'][0]).tolist()) <= {0., 1.}


def test_alignment_dataset_does_not_require_textures(tmp_path):
    from PIL import Image
    from data.grain import GrainDataset
    folder=tmp_path/'paddy'/'train'/'good';folder.mkdir(parents=True)
    Image.new('RGB',(160,32),(90,90,90)).save(folder/'a.png')
    ds=GrainDataset(str(tmp_path),classname='paddy',five_split=True,resize=[32,32],imagesize=[32,32],padding=0,anomaly_source_path='',generate_anomalies=False)
    item=ds[0]
    assert len(item['image'])==5
    assert all(torch.equal(a,b) for a,b in zip(item['image'],item['aug_image']))
    assert all(m.sum()==0 for m in item['aug_mask'])


def test_image_only_evaluation_does_not_accumulate_maps():
    from models.simplenet import SimpleNet
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    class Images(Dataset):
        with_mask=False
        def __len__(self):return 2
        def __getitem__(self,i):
            return dict(image=[torch.zeros(3,8,8)],mask=[torch.zeros(1,8,8)],is_anomaly=i,image_path=str(i))
    model=SimpleNet(torch.device('cpu'));model.forward_modules=nn.ModuleDict();model.visualization=False;model.add_ica=False
    model._predict=lambda x: ([0.2], [np.zeros((8,8))], [], x[0], [])
    scores,maps,_,labels,masks=model.predict(DataLoader(Images(),batch_size=1))
    assert len(scores)==2 and labels==[0,1] and maps==[] and masks==[]


def test_worker_initialization_restores_imgaug_cache():
    import imgaug.augmenters as iaa
    from data.grain import initialize_worker
    iaa.AddToHueAndSaturation._LUT_CACHE=None
    initialize_worker(0)
    assert iaa.AddToHueAndSaturation._LUT_CACHE is not None
