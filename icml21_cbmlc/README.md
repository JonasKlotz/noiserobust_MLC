##### CbMLC

This README file provides instructions to run CbMLC.

1) To obtain the datasets other than Pascal VOC, follow the commands at https://github.com/QData/LaMP
   For Pascal VOC, our code downloads it automatically.

2) To generate initial g(.), run ``python get_multiword2vec.py dataset_name``
   Running this requires to have a local copy of GloVe, which can be accessed at https://nlp.stanford.edu/projects/glove/

3) To generate noisy datasets, run ``python add_noise.py dataset_name noise_level noise_type``

4) Finally, to run CbMLC, here is an example script 

   ```CUDA_VISIBLE_DEVICES=0 python main.py -dataset bibtext_0.5_uniform-positive -batch_size 32 -d_model 300 -n_layers_enc 2 -n_layers_dec 2 -n_head 4 -epoch 50 -dropout 0.200000 -dec_dropout 0.200000 -lr 0.000200 -encoder graph -decoder graph -int_preds -lr_decay 1.000000 -lr_step_size 12 -loss asl -asl_pg 4.000000 -asl_ng 4.000000 -asl_eps 0.000000 -asl_clip 0.100000 -load_tgt_embedding -tgt_emb_l2 0.000000```

   -dataset specifies the dataset (for clean datasets, it is {bibtext, reuters, delicious, rcv1}; for the noisy datasets, it is {bibtext, reuters, delicious, rcv1}_{0.01, ..., 0.5}_{combined, uniform, uniform-positive, one-postive}
   for Pascal VOC, it is a bit different - for both clean and noisy settings, it is ``-dataset voc``, but please also add ``-noisy_level xxx -noisy_label xxx'
   -batch size spcifies the batch size
   -d_model specifies the size of latent vector. If using GloVE, this has to be 300
   -n_layers_dec and -n_layers_dec specify the # of layers for encoders when using transformer and for decoders
   -n_head specifies # of attention heads
   -epoch specifies # of epochs to run
   -dropout specifies dropout in encoder when using transformer
   -dec_dropout specifies dropout in decoder
   -lr specifies learning rate
   -int_preds specifies whether to compute loss at intermediate layers
   -lr_decay specifies how much to decay on the learning rate, setting it as 1 means no learning rate decaying
   -lr_step_size specifies how many steps before one decay
   -loss specifies loss, which can either be bce or asl
   -asl_pg -asl_ng -asl_eps -asl_clip these four flags are for using the asl loss, pg is gamma+, ng is gamma-, eps is to avoid zero division, and clip is m
   -load_tgt_embedding is to load pretrained word embeddings
   -tgt_emb_l2 the coefficient for context-based regularization
