import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
import pandas as pd
import numpy as np

import category_encoders as ce


# Structure is from jodie.py, the dataset used in the tgn.py example

class NIDSToNIoTDataset(InMemoryDataset):
    url = "https://cloudstor.aarnet.edu.au/plus/s/ds5zW91vdgjEj9i/download?path=%2FTrain_Test_datasets%2FTrain_Test_Network_dataset&files=Train_Test_Network.csv"
    names = ["train_test_network"]

    def __init__(
        self,
        root: str,
        name: str,
        train_size: int = 100000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in self.names
        self.train_size = train_size
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        #download_url(self.url, self.raw_dir, filename=self.raw_paths[0])
        print("Download the dataset manually")

    def from_csv_to_graph(self, df):

        nodes = df['src_ip'].unique().tolist()
        print("Unique src ip: ", len(nodes))
        nodes.extend(df['dst_ip'].unique().tolist())
        print("Unique dst ip: ", len(df['dst_ip'].unique()))
        nodes = list(set(nodes))
        print("Unique nodes: ", len(nodes))
        print("Count of Nodes which are included in src and dst: ",len(df['src_ip'].unique()) + len(df['dst_ip'].unique()) - len(nodes))
        nodes = pd.DataFrame(nodes, columns=['node'])   
        nodes['node_id'] = nodes.index
        edges = pd.DataFrame()
        edges = df
        print(edges.shape)
        edges = edges.merge(nodes, left_on='src_ip', right_on='node', how='left')
        edges = edges.rename(columns={'node_id': 'src_id'})
        edges = edges.drop(columns=['node'])
        edges = edges.merge(nodes, left_on='dst_ip', right_on='node', how='left')
        edges = edges.rename(columns={'node_id': 'dst_id'})
        edges = edges.drop(columns=['node'])
        #edges = edges.sort_values(by=['ts']) # was was the error, because the index was not sorted - so i had 3086 lines wrong for evaluation 
        #edges['ts'] = edges['ts'] - edges['ts'].min()
        edges["ts"] = edges.index  # use index as timestamp
        src = edges['src_id'].values 
        dst = edges['dst_id'].values
        #t = edges["ts"].values
        t = edges.index.values
        label = edges['label'].values
        msg = edges

        #return pd.DataFrame({"u": src, "i": dst, "ts": t, "idx": idx, "w": w, "label":label}), msg, None
        return src, dst, t, label, msg

    def process(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(self.raw_paths[0])
        tmp_src, tmp_dst, tmp_t, tmp_y, tmp_msg = self.from_csv_to_graph(df)
        
        neg_src, neg_dst, neg_t, neg_y, neg_msg = self.negativ_sampling(tmp_src, tmp_dst, tmp_t, tmp_y, tmp_msg)

        scaler = StandardScaler()
        encoder = ce.OneHotEncoder()
        src, dst, t, y, msg = self.parse_to_tensor(scaler, encoder, tmp_src, tmp_dst, tmp_t, tmp_y, tmp_msg)
        neg_src, neg_dst, neg_t, neg_y, neg_msg = self.parse_to_tensor(scaler, encoder, neg_src, neg_dst, neg_t, neg_y, neg_msg)

        # care about nans 
        msg = np.nan_to_num(msg)

        src = torch.from_numpy(src).to(torch.long)
        dst = torch.from_numpy(dst).to(torch.long)
        t = torch.from_numpy(t).to(torch.long)
        y = torch.from_numpy(y).to(torch.long)
        # msg = torch.from_numpy(msg).to(torch.float)
        msg = torch.from_numpy(msg).to(torch.float) # create dummy msg


        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

        neg_src = torch.from_numpy(neg_src).to(torch.long)
        neg_dst = torch.from_numpy(neg_dst).to(torch.long)
        neg_t = torch.from_numpy(neg_t).to(torch.long)
        neg_y = torch.from_numpy(neg_y).to(torch.long)
        # neg_msg = torch.from_numpy(neg_msg).to(torch.float)
        neg_msg = torch.from_numpy(neg_msg).to(torch.float) # create dummy msg


        neg_data = TemporalData(src=neg_src, dst=neg_dst, t=neg_t, msg=neg_msg, y=neg_y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data, neg_data], self.processed_paths[0])



    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'


##############################################################################################################
#     Negative Sampling
##############################################################################################################

    def negativ_sampling(self, src, dst, t, y, msg):
        # orientate at normal behavior
        # df = df[df['label'] == 0]
        # # Copy the original DataFrame
        # negative_samples = df.copy()

        # # Shuffle values in categorical columns
        # for col in df.select_dtypes(include='object').columns:
        #     negative_samples[col] = np.random.permutation(df[col].values)

        # # Shuffle values in numeric columns
        # for col in df.select_dtypes(include='number').columns:
        #     negative_samples[col] = np.random.permutation(df[col].values)

        # # get length of negative samples and y - fill the missing rows in negative sample with random rows form negative samples
        # missing_rows = len(y) - len(negative_samples) # isnt working well at the moment
        # print("missing rows: ", missing_rows) # isnt working well at the moment
        # negative_samples = negative_samples.append(negative_samples.sample(missing_rows)) # isnt working well at the moment

        # Set label to 1
        y = np.ones(len(y))
        src = np.random.permutation(src)
        dst = np.random.permutation(dst)

        msg = msg 

        return src, dst, t, y, msg

##############################################################################################################
#     Utility
##############################################################################################################

    def parse_to_tensor(self, scaler, encoder, src, dst, t, y, df):
            #numeric values
            numeric_values = df.select_dtypes(include=['int64', 'float64'])
            numeric_values.drop(['ts', 'src_port', 'dst_port', 'label'], axis=1, inplace=True) # 'label' should also be dropped, but we need it here to be used later as label 
            numeric_columns = numeric_values.columns
            #category values
            category_values = df.select_dtypes(include=['object'])
            category_values.drop(['src_ip', 'dst_ip', 'weird_name', 'weird_addl', 'weird_notice', 'type'], axis=1, inplace=True)
            category_values.drop(['dns_query', 'http_trans_depth', 'ssl_subject', 'ssl_issuer', 'http_uri', 'http_user_agent' ], axis=1, inplace=True)
            category_columns = category_values.columns 
            category_values = df[category_columns].astype('str')
            print(category_columns)
            train_set_len = self.train_size
            val_set_len = 25000 # so test can begin at 150000
            #df_cat = pd.get_dummies(category_values.astype('str'), columns=category_columns, prefix="cat_dummies")  
            #print("df_cat shape: ", df_cat.shape)
            numeric_df = df[numeric_columns]
            from sklearn.utils.validation import check_is_fitted
            try:
                check_is_fitted(scaler)
                scaled_numeric_df_train = pd.DataFrame(scaler.transform(numeric_df.iloc[:train_set_len]), columns=numeric_columns)
                df_cat = pd.DataFrame(encoder.transform(df_cat.iloc[:train_set_len]))
            except:
                df_cat = pd.DataFrame(encoder.fit_transform(category_values.iloc[:train_set_len]))
                scaled_numeric_df_train = pd.DataFrame(scaler.fit_transform(numeric_df.iloc[:train_set_len]), columns=numeric_columns)
            #scaled_numeric_df_val_and_test = pd.DataFrame(scaler.transform(numeric_df.iloc[(train_set_len - val_set_len):]), columns=numeric_columns)
            scaled_numeric_df_val_and_test = pd.DataFrame(scaler.transform(numeric_df.iloc[(train_set_len):]), columns=numeric_columns)
            whole_df = pd.concat([scaled_numeric_df_train, scaled_numeric_df_val_and_test], axis=0)
            whole_df.reset_index(drop=True, inplace=True)
            df_num = whole_df[numeric_columns]
            df_msg  = pd.concat([df_num, df_cat], axis=1)
            
            msg = df_msg.values 
            print("msg shape: ", msg.shape)
            return src, dst, t, y, msg