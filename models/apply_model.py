import pandas as pd
import numpy as np
import pickle
import patsy

class Model(object):

    #Model: object that contain a model, either ReachSb or ReachSbPb

    def __init__(self, use_ad_type=False):

        #initializes a model object
        #inputs:
        #   use_ad_type: (optional) Use the ReachSb (use_ad_type=False) or the
        #   ReachSbPb model (use_ad_type=True). Default is False.

        if use_ad_type == False:
            self.model_type_ = 'without_ad_type'

            file = open("ReachSb.p","rb")
            model_info = pickle.load(file)
            file.close()

            self.model_= model_info['model']
            self.formula_= model_info['formula']
            self.train_data_ = model_info['train_data']

        else:
            self.model_type_ = 'with_ad_type'

            file = open("ReachSbPb.p","rb")
            model_info = pickle.load(file)
            file.close()

            self.model_ = model_info['model']
            self.formula_ = model_info['formula']
            self.train_data_ = model_info['train_data']

    def make_prediction(self, reach=[0], site=['All 4'], ad_type=None):

        # Make a prediction. The prediction will be made over a grid that
        # relates each predictor.
        #
        # Inputs:
        #   reach: List of floats. Values for reach per day to be used for the
        #          prediction. Default is [0]
        #   site: List of strings. Name of media to be used for the prediction.
        #         Default is 'All 4'.
        #   ad_type: (optional) List of strings or None. Name of the ad types
        #            to be used for the prediction. Default is None.
        #
        # Returns: Pandas Dataframe with the Reach, Site, Ad Type and predicted
        #         total conversions per day. Reach and total conversions are in
        #         normal space (not log).

        # apply encoding
        _,train_data = patsy.dmatrices(self.formula_,
                                       self.train_data_,
                                       return_type='dataframe')

        encoding = train_data.design_info

        # tranform reach to log10 scale
        reach = np.log10(np.array(reach)+1)

        predictions = []
        if self.model_type_ == 'without_ad_type':

            #build prediction dataframe
            new_reach = []
            new_site = []
            for r in reach:
                for s in site:
                    new_reach.append(r)
                    new_site.append(s)

            new_df = pd.DataFrame({'Reach':new_reach, 'Sb':new_site})
            new_df = new_df.drop_duplicates()
            self.new_df_ = new_df

        elif self.model_type_ == 'with_ad_type':

            #build prediction dataframe
            new_reach = []
            new_site = []
            new_ad_type = []
            for r in reach:
                for s in site:
                    for ad in ad_type:
                        new_reach.append(r)
                        new_site.append(s)
                        new_ad_type.append(ad)

            new_df = pd.DataFrame({'Reach':new_reach, 'Sb':new_site, 'Pb':new_ad_type})
            new_df = new_df.drop_duplicates()
            self.new_df_ = new_df

        new_df_enc, = patsy.build_design_matrices([encoding],self.new_df_,
                                                return_type='dataframe')
        predictions = self.model_.predict(new_df_enc)

        # total conversions and reach in normal scale
        predictions = 10**predictions - 1
        self.new_df_['Reach'] = 10**self.new_df_['Reach']-1

        predictions_df = pd.DataFrame(predictions, columns=['Conversions/day'])

        new_df = pd.concat([self.new_df_, predictions_df], axis=1)
        new_df = new_df.sort_values(by=['Conversions/day'], ascending=False)

        if self.model_type_ == 'without_ad_type':
            new_df.rename(columns={'Reach':'Reach/day','Sb':'Site'},
                           inplace=True)
        elif self.model_type_ == 'with_ad_type':
            new_df.rename(columns={'Reach':'Reach/day','Sb':'Site',
                                         'Pb':'Ad Type'},
                           inplace=True)

        return new_df
