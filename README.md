# IDL_project

Summary of files and intended behaviour (look inside model components folder for the files)
1. positional_encoding.py: python file with the time series positional encoding for the financial data (currently the same as HW2 PE)
2. timeseries_embedding.py: Time2VecTorch class, which is the embedding fo the time dimension of data into a higher dimension (governed by d_model) using linear and periodic layers. 
3. transformers.py: has the transformer architecture, based on hw. decoder only. uses the same sublayers coded from the hw.
4. data_reader: constructs financial data using yfinance. chops the data into window legth, and concatenates subsequent windows to make one large data of size (B, N_w*S_w, N_f)---> (Batch size, num_windows*window size, num_features)

[ raw financial features ] ──► [ Feature Projection Layer (MLP or Linear) ]
                            │
[ time feature ] ───────────► [ Time2Vec ]
                            │
            └──────────────► [ CONCATENATE ] ──► [ rest of your model ]

# TODO:
1. Implement a ipynb with the following:
    1.1: reads the dataset using dataloader (specify batch_size)
    1.2: creates the transformer model 
    1.3: visualizes embeddings, encodings, etc (optional/ later)
    1.4: trains the model and generates financial predictions for x timesteps ahead.