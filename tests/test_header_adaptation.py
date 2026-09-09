import csv
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from src.header_ocr import tighten_header, load_header_recognizer
from scripts.prepare_real_header_dataset import store_sample
from scripts.train_header_ocr import verified_records, split_real, initialize_model
from main import _complete_plate_text
from test_plate_full_text import fixture, HeaderRecognizer


class HeaderAdaptationTests(unittest.TestCase):
    def test_smoke_or_unqualified_header_checkpoint_not_loaded(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'best.pt'
            for config in [{'task':'header_finetune','smoke_test':True},
                           {'task':'header_finetune','production_ready':False}]:
                torch.save({'config':config},path)
                with self.assertRaises(ValueError):load_header_recognizer(path,'cpu')

    def test_recovery_rotation_and_dataset_mismatch_guard(self):
        import torch, random, numpy as np
        from scripts.train_ocr import save_ocr_recovery, load_ocr_recovery
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config={'vocab':'AB12','target_epochs':3,'real_fingerprint':'verified-data'}
            checkpoint={'epoch':1,'config':config,'vocab':'AB12','model_state_dict':{'w':torch.ones(1)},
                'optimizer_state_dict':{},'scheduler_state_dict':{},'best_exact_accuracy':.5,'best_cer':.5,
                'python_random_state':random.getstate(),'numpy_random_state':np.random.get_state(),
                'torch_random_state':torch.get_rng_state()}
            save_ocr_recovery(checkpoint,root,True)
            checkpoint['epoch']=2
            save_ocr_recovery(checkpoint,root,False)
            self.assertEqual(load_ocr_recovery(root,config,True)[2],2)
            self.assertEqual(torch.load(root/'previous.pt',weights_only=False)['epoch'],1)
            self.assertEqual(torch.load(root/'best.pt',weights_only=False)['epoch'],1)
            with self.assertRaises(RuntimeError):
                load_ocr_recovery(root,config|{'real_fingerprint':'changed'},True)

    def test_trim_separated_bolts_preserves_text_band(self):
        im=Image.new('RGB',(300,50),'white'); draw=ImageDraw.Draw(im)
        draw.ellipse((3,12,20,30),fill='black');draw.ellipse((280,12,297,30),fill='black')
        for x in range(65,230,12):draw.rectangle((x,10,x+7,38),fill='black')
        tight,info=tighten_header(im)
        self.assertGreater(info['bounds'][0],20)
        self.assertLess(info['bounds'][2],280)
        self.assertLessEqual(info['bounds'][0],65)
        self.assertGreaterEqual(info['bounds'][2],229)

    def test_blank_crop_is_unchanged(self):
        im=Image.new('RGB',(100,25),'white')
        self.assertEqual(tighten_header(im)[0].size,im.size)

    def test_manual_labels_and_unlabeled_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'dataset'; source=Path(temp)/'source.png'
            im=Image.new('RGB',(120,32),'white');im.save(source)
            unlabeled=store_sample(root,im,source)
            self.assertEqual(unlabeled.parent.name,'unlabeled')
            self.assertEqual(verified_records(root)[0],[])
            label='परिवहन XY १२'
            labeled=store_sample(root,im,source,label)
            records,groups,_=verified_records(root)
            self.assertEqual(len(records),1)
            self.assertEqual(records[0].label,label)
            with self.assertRaisesRegex(ValueError,'Need 100'):
                split_real(records,groups)
            labeled.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Image changed'):
                verified_records(root)

    def test_smoke_requires_verified_sample(self):
        with self.assertRaisesRegex(ValueError,'No manually verified'):
            split_real([],{},True)

    def test_missing_header_checkpoint_falls_back(self):
        self.assertIsNone(load_header_recognizer(Path('/tmp/no-such-header-checkpoint-test.pt'),'cpu'))

    def test_header_model_cannot_change_registration(self):
        result=fixture()
        registration=(result.text,result.ocr_confidence,result.final_status,result.validation_status)
        models=tuple(HeaderRecognizer([]) for _ in range(3))
        header=HeaderRecognizer(['परिवहन'])
        _complete_plate_text(result,models,.8,False,header)
        self.assertEqual(result.header_text,'परिवहन')
        self.assertEqual([m.calls for m in models],[0,0,0])
        self.assertEqual((result.text,result.ocr_confidence,result.final_status,result.validation_status),registration)

    def test_checkpoint_transfer_and_one_in_memory_gradient_step(self):
        import torch
        from src.utils import PROJECT_ROOT
        from src.recognizer import PARSeqRecognizer
        path=PROJECT_ROOT/'models/ocr_mixed_v2/best.pt'
        if not path.exists(): self.skipTest('Mixed-v2 checkpoint not installed')
        source=torch.load(path,map_location='cpu',weights_only=False)
        model=initialize_model(source,source['vocab'])
        for key,value in source['model_state_dict'].items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]))
        # Synthetic test fixture with an explicitly rendered label, never an OCR pseudo-label.
        im=Image.new('RGB',(128,32),'white');ImageDraw.Draw(im).text((20,8),'AB12',fill='black')
        adapter=PARSeqRecognizer.__new__(PARSeqRecognizer)
        adapter.torch,adapter.device,adapter.input_normalization=torch,'cpu','doctr_parseq'
        model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        loss=model(adapter._tensor(im),target=['AB12'])['loss']
        self.assertTrue(torch.isfinite(loss));loss.backward();optimizer.step()
        del model,optimizer
        vocab=source['vocab']+'ह'
        expanded=initialize_model(source,vocab)
        self.assertTrue(torch.equal(expanded.state_dict()['head.weight'][:len(source['vocab'])],source['model_state_dict']['head.weight'][:-1]))
        self.assertTrue(torch.equal(expanded.state_dict()['embed.embedding.weight'][len(vocab):],source['model_state_dict']['embed.embedding.weight'][len(source['vocab']):]))


if __name__=='__main__':unittest.main()
