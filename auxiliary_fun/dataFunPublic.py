import scipy.io as scio
import numpy as np
import os

def readPublicData():
    dataPath='.\\open_source _datasets\\'

    CGL_nir_data = scio.loadmat(os.path.join(dataPath, 'CGL_nir.mat'))
    corn_data = scio.loadmat(os.path.join(dataPath, 'corn.mat'))
    nir_shootout_2002_data = scio.loadmat(os.path.join(dataPath, 'nir_shootout_2002.mat'))
    BP50GA_data = scio.loadmat(os.path.join(dataPath, 'SWRI_Diesel_NIR/BP50GATEST.mat'))
    CNGA_data = scio.loadmat(os.path.join(dataPath, 'SWRI_Diesel_NIR/CNGATEST.mat'))
    D4052GA_data = scio.loadmat(os.path.join(dataPath, 'SWRI_Diesel_NIR/D4052GATEST.mat'))
    FREEZEGA_data = scio.loadmat(os.path.join(dataPath, 'SWRI_Diesel_NIR/FREEZEGATEST.mat'))
    TOTALGA_data = scio.loadmat(os.path.join(dataPath, 'SWRI_Diesel_NIR/TOTALGATEST.mat'))
    VISCGA_data = scio.loadmat(os.path.join(dataPath, 'SWRI_Diesel_NIR/VISCGATEST.mat'))
    
    publicData = {'CGL_nir':
                    {'casein':
                        {'train': [CGL_nir_data['Xcal']['data'][0][0], 
                                CGL_nir_data['Ycal']['data'][0][0][:,0]],
                        'vlide': [CGL_nir_data['Xtest']['data'][0][0],
                                CGL_nir_data['Ytest']['data'][0][0][:,0]]
                        },
                    'glucose':
                        {'train': [CGL_nir_data['Xcal']['data'][0][0], 
                                CGL_nir_data['Ycal']['data'][0][0][:,1]],
                        'vlide': [CGL_nir_data['Xtest']['data'][0][0],
                                CGL_nir_data['Ytest']['data'][0][0][:,1]]
                        },
                    'lactate':
                        {'train': [CGL_nir_data['Xcal']['data'][0][0], 
                                CGL_nir_data['Ycal']['data'][0][0][:,2]],
                        'vlide': [CGL_nir_data['Xtest']['data'][0][0],
                                CGL_nir_data['Ytest']['data'][0][0][:,2]]
                        },
                    'moisture':
                        {'train': [CGL_nir_data['Xcal']['data'][0][0], 
                                CGL_nir_data['Ycal']['data'][0][0][:,3]],
                        'vlide': [CGL_nir_data['Xtest']['data'][0][0],
                                CGL_nir_data['Ytest']['data'][0][0][:,3]]
                        },
                    },
                'corn':
                    {'oil':
                        {'train': [corn_data['m5spec']['data'][0][0][:60],
                                corn_data['propvals']['data'][0][0][:60,1]],
                        'vlide': [corn_data['m5spec']['data'][0][0][60:], 
                                corn_data['propvals']['data'][0][0][60:,1]]
                        },
                    'protein':
                        {'train': [corn_data['m5spec']['data'][0][0][:60],
                                corn_data['propvals']['data'][0][0][:60,2]],
                        'vlide': [corn_data['m5spec']['data'][0][0][60:], 
                                corn_data['propvals']['data'][0][0][60:,2]]
                        },
                    'starch':
                        {'train': [corn_data['m5spec']['data'][0][0][:60],
                                corn_data['propvals']['data'][0][0][:60,3]],
                        'vlide': [corn_data['m5spec']['data'][0][0][60:], 
                                corn_data['propvals']['data'][0][0][60:,3]]
                        },
                    },
                'nir_shootout_2002':
                    {'attr1':
                        {'train': [ np.vstack((nir_shootout_2002_data['test_1']['data'][0][0],
                                         nir_shootout_2002_data['calibrate_1']['data'][0][0])),
                                    np.vstack((nir_shootout_2002_data['test_Y']['data'][0][0],
                                         nir_shootout_2002_data['calibrate_Y']['data'][0][0]))[:,0]],
                        'vlide': [ nir_shootout_2002_data['validate_1']['data'][0][0], 
                                   nir_shootout_2002_data['validate_Y']['data'][0][0][:,0]]},
                    'attr2':
                        {'train': [ np.vstack((nir_shootout_2002_data['test_1']['data'][0][0],
                                         nir_shootout_2002_data['calibrate_1']['data'][0][0])),
                                    np.vstack((nir_shootout_2002_data['test_Y']['data'][0][0],
                                         nir_shootout_2002_data['calibrate_Y']['data'][0][0]))[:,1]],
                        'vlide': [ nir_shootout_2002_data['validate_1']['data'][0][0], 
                                   nir_shootout_2002_data['validate_Y']['data'][0][0][:,1]]},
                    'attr3':
                        {'train': [ np.vstack((nir_shootout_2002_data['test_1']['data'][0][0],
                                         nir_shootout_2002_data['calibrate_1']['data'][0][0])),
                                    np.vstack((nir_shootout_2002_data['test_Y']['data'][0][0],
                                         nir_shootout_2002_data['calibrate_Y']['data'][0][0]))[:,2]],
                        'vlide': [ nir_shootout_2002_data['validate_1']['data'][0][0], 
                                   nir_shootout_2002_data['validate_Y']['data'][0][0][:,2]]},
                    },
                'SWRI_Diesel_NIR':
                    {'BP50GA':
                        {'train': [ np.vstack((BP50GA_data['bp50_s1d_hl'], BP50GA_data['bp50_s1d_ll_a'])),
                                    np.vstack((BP50GA_data['bp50_y1_hl'], BP50GA_data['bp50_y1_ll_a']))[:,0]],
                        'vlide': [ BP50GA_data['bp50_s1d_ll_b'], BP50GA_data['bp50_y1_ll_b'][:,0]]
                        },
                     'CNGA':
                        {'train': [np.vstack((CNGA_data['cn_sd_hl'], CNGA_data['cn_sd_ll_a'])),
                                   np.vstack((CNGA_data['cn_y_hl'], CNGA_data['cn_y_ll_a']))[:,0]],
                         'vlide': [CNGA_data['cn_sd_ll_b'], CNGA_data['cn_y_ll_b'][:,0]]
                        },
                     'D4052GA':
                        {'train': [np.vstack((D4052GA_data['d_sd_hl'], D4052GA_data['d_sd_ll_a'])),
                                    np.vstack((D4052GA_data['d_y_hl'], D4052GA_data['d_y_ll_a']))[:,0]],
                        'vlide': [D4052GA_data['d_sd_ll_b'], D4052GA_data['d_y_ll_b'][:,0]]
                        },
                     'FREEZEGA':
                        {'train': [np.vstack((FREEZEGA_data['f_sd_hl'], FREEZEGA_data['f_sd_ll_a'])),
                                    np.vstack((FREEZEGA_data['f_y_hl'], FREEZEGA_data['f_y_ll_a']))[:,0]],
                        'vlide': [FREEZEGA_data['f_sd_ll_b'], FREEZEGA_data['f_y_ll_b'][:,0]]
                        },
                     'TOTALGA':
                        {'train': [np.vstack((TOTALGA_data['t_sd_hl'], TOTALGA_data['t_sd_ll_a'])),
                                    np.vstack((TOTALGA_data['t_y_hl'], TOTALGA_data['t_y_ll_a']))[:,0]],
                        'vlide': [TOTALGA_data['t_sd_ll_b'], TOTALGA_data['t_y_ll_b'][:,0]]
                        },
                     'VISCGA':
                        {'train': [np.vstack((VISCGA_data['v_sd_hl'], VISCGA_data['v_sd_ll_a'])),
                                    np.vstack((VISCGA_data['v_y_hl'], VISCGA_data['v_y_ll_a']))[:,0]],
                        'vlide': [VISCGA_data['v_sd_ll_b'], VISCGA_data['v_y_ll_b'][:,0]]
                        }
                    },                  
                }
    
    return publicData
