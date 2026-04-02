import logging
import os
from abc import abstractmethod

import torch
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx

class BaseTester(object):
    def __init__(self, model,loss_fns, m_metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_fns = loss_fns
        self.m_metric_ftns = m_metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        self.dataset_name = self.args.dataset_name
        self.record_dir = self.args.record_dir

        self.model_name = self.args.model_name

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, loss_fns, m_metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, loss_fns, m_metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        records = []
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, batch in enumerate(self.test_dataloader):
                ids = batch['exam_ids']
                images = batch['images'].to(self.device)
                mask = batch['mask'].to(self.device)
                phases = batch['phases'].to(self.device)
                locations = batch['locations'].to(self.device)
                lesions = batch['lesions'].to(self.device)
                diagnosis = batch['diagnosis'].to(self.device)
                phase_logits, location_logits, lesion_logits, diagnosis_logits, confidence, normalized_entropy = self.model(images, mask)


                if self.model_name in ["MultiTaskTransformerGNN", "MultiTaskTransformerCatGNN"]:
                    phase_ids = phase_logits.argmax(dim=-1)         
                    location_ids = location_logits.argmax(dim=-1)    
                    lesion_scores = torch.sigmoid(lesion_logits)     
                    diagnosis_ids = diagnosis_logits.argmax(dim=-1)  
                    diagnosis_scores = torch.softmax(diagnosis_logits, dim=-1) 


                    for i in range(images.size(0)):  
                        exam_id = batch['exam_ids'][i]

                        sample_phase_ids = phase_ids[i]              
                        sample_location_ids = location_ids[i]        
                        sample_lesion_scores = lesion_scores[i]    
                        sample_mask = mask[i].unsqueeze(0)          

                        graph_data = self.model.module._build_graph(
                        # graph_data = self.model._build_graph(
                            sample_phase_ids,
                            sample_location_ids,
                            sample_lesion_scores,
                            B=1,
                            T=sample_phase_ids.size(0),
                            mask=sample_mask
                        )

                        if graph_data is not None:
                            nx_graph = to_networkx(graph_data, to_undirected=True)
                            node_types = graph_data.node_type
                            node_values = graph_data.node_value

                            for node, _ in nx_graph.nodes(data=True):
                                if node in node_types:
                                    nx_graph.nodes[node]['type'] = node_types[node]
                                    nx_graph.nodes[node]['value'] = node_values[node]
                                    nx_graph.nodes[node]['time_idx'] = int(graph_data.time_idx[node].item())

                            nx_graph.graph['diagnosis'] = diagnosis_ids[i].item()
                            # nx_graph.graph['diagnosis_score'] = diagnosis_scores[i, diagnosis_ids[i]].item()
                            nx_graph.graph['confidence'] = float(confidence[i].item())
                            nx_graph.graph['normalized_entropy'] = float(normalized_entropy[i].item())

                            save_path = os.path.join(self.save_dir, f'{exam_id}.gml')
                            nx.write_gml(nx_graph, save_path)



                test_res.extend(diagnosis_logits)
                test_gts.extend(diagnosis)


                pred_phase = torch.argmax(phase_logits, dim=-1)              
                pred_location = torch.argmax(location_logits, dim=-1)        
                pred_lesion = (torch.sigmoid(lesion_logits) > 0.5).int()   
                pred_diagnosis = torch.argmax(diagnosis_logits, dim=-1)    

                lesion_prob = torch.sigmoid(lesion_logits)   
                diagnosis_probs = torch.softmax(diagnosis_logits, dim=-1) 


                B, T = phases.shape

                for b in range(B):
                    id = ids[b]
                    true_diag = int(diagnosis[b].item())
                    pred_diag = int(pred_diagnosis[b].item())


                    score_dict = {
                        f"score_diagnosis_{i}": round(float(diagnosis_probs[b, i].item()), 4)
                        for i in range(diagnosis_probs.size(1))
                    }

                    for t in range(T):
                        if mask[b, t] == 0:
                            continue

                        record = {
                            "no_id": id,
                            "image_index": t,
                            "true_diagnosis": true_diag,
                            "pred_diagnosis": pred_diag,
                            "confidence": round(float(confidence[b].item()), 4),          
                            "normalized_entropy": round(float(normalized_entropy[b].item()), 4),  
                            **score_dict,  
                            "true_phase": int(phases[b, t].item()),
                            "pred_phase": int(pred_phase[b, t].item()),
                            "true_location": int(locations[b, t].item()),
                            "pred_location": int(pred_location[b, t].item()),
                        }

                        for i in range(7):
                            record[f"true_lesion_{i}"] = int(lesions[b, t, i].item())
                            record[f"pred_lesion_{i}"] = int(pred_lesion[b, t, i].item())
                            record[f"score_lesion_{i}"] = round(float(lesion_prob[b, t, i].item()), 4)

                        records.append(record)

            test_met = self.m_metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                       {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})      


        df = pd.DataFrame(records)
        df.to_csv(self.save_dir + '/pre_outputs.csv', index=False)

        print(log)
        
        return log
