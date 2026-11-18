"""Auxiliary DAgger Adam over detached Teacher labels, independent of PPO."""
import torch
from .tabletop_stereo_bc import StereoBCTrainer


class StereoDAggerTrainer(StereoBCTrainer):
    def train(self,dataset,*,batch_size=32,epochs=1,generator=None):
        if type(batch_size) is not int or batch_size<1 or type(epochs) is not int or epochs<1:
            raise ValueError('positive DAgger epochs and minibatch size required')
        if not len(dataset):raise ValueError('nonempty independently aggregated visited-state dataset required')
        reports=[]
        for epoch in range(epochs):
            order=torch.randperm(len(dataset),generator=generator).tolist()
            for start in range(0,len(order),batch_size):
                report=self.step(dataset.batch(order[start:start+batch_size]))
                report['epoch']=epoch+1;reports.append(report)
        return reports
