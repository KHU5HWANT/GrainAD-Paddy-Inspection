import torch
from utils.training_state import save_training,load_training,pause_requested

def test_optimizer_resume_matches_next_update(tmp_path):
    torch.manual_seed(4)
    a=torch.nn.Linear(3,1);opt=torch.optim.Adam(a.parameters(),lr=.01)
    x=torch.randn(4,3)
    def update(m,o):
        o.zero_grad();m(x).square().mean().backward();o.step()
    update(a,opt)
    path=tmp_path/'state.pth';save_training(path,model=a.state_dict(),optimizer=opt.state_dict(),epoch=3)
    b=torch.nn.Linear(3,1);ob=torch.optim.Adam(b.parameters(),lr=.01)
    state=load_training(path,'cpu');b.load_state_dict(state['model']);ob.load_state_dict(state['optimizer'])
    assert state['epoch']==3
    update(a,opt);update(b,ob)
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb)

def test_pause_flag(tmp_path,monkeypatch):
    flag=tmp_path/'pause.request';monkeypatch.setenv('GRAINAD_PAUSE_FILE',str(flag))
    assert not pause_requested();flag.touch();assert pause_requested()

def test_failed_save_preserves_previous_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'state.pth';save_training(path,epoch=1)
    original=path.read_bytes()
    def fail(*args,**kwargs):raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch,'save',fail)
    import pytest
    with pytest.raises(OSError):save_training(path,epoch=2)
    assert path.read_bytes()==original

def test_grainad_pause_and_resume_continues_epoch(tmp_path,monkeypatch):
    from models.simplenet import SimpleNet
    def make(folder):
        torch.manual_seed(3)
        m=SimpleNet(torch.device('cpu'));m.ckpt_dir=str(folder);folder.mkdir(exist_ok=True)
        m.discriminator=torch.nn.Linear(2,1);m.dsc_opt=torch.optim.Adam(m.discriminator.parameters(),lr=.01)
        m.dsc_schl=torch.optim.lr_scheduler.CosineAnnealingLR(m.dsc_opt,2)
        m.pre_proj=0;m.train_backbone=False;m.meta_epochs=2
        def step(data):
            m.dsc_opt.zero_grad();m.discriminator(torch.ones(2,2)).square().mean().backward();m.dsc_opt.step()
        m._train_discriminator=step;m.predict=lambda data:([],[],[],[],[])
        m._evaluate=lambda *args:(.5,-1,-1)
        return m
    flag=tmp_path/'pause';monkeypatch.setenv('GRAINAD_PAUSE_FILE',str(flag))
    baseline=make(tmp_path/'baseline');baseline.train([],[],False)
    first=make(tmp_path/'resumed');flag.touch()
    import pytest
    with pytest.raises(SystemExit) as exc:first.train([],[],False)
    assert exc.value.code==75
    assert load_training(tmp_path/'resumed'/'training_state.pth','cpu')['epoch']==1
    flag.unlink();second=make(tmp_path/'resumed');second.train([],[],False)
    assert load_training(tmp_path/'resumed'/'training_state.pth','cpu')['epoch']==2
    for a,b in zip(baseline.discriminator.parameters(),second.discriminator.parameters()):torch.testing.assert_close(a,b)
