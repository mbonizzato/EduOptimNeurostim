import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--run1', '-r1', type=str, default=None, required=True)
args = parser.parse_args()

sbs_mapped_arr = np.load(os.path.join(args.run1,'sbs_mapped_arr.npy'))
R_mean = np.load(os.path.join(args.run1,'Rmean.npy'))
ch2xy = np.load(os.path.join(args.run1,'ch2xy.npy'))
sbs_save_query = np.load(os.path.join(args.run1,'sbs_save_query.npy'))
sbs_query_idx = np.load(os.path.join(args.run1,'sbs_query_idx.npy'))
sbs_best_query = np.load(os.path.join(args.run1,'sbs_best_query.npy'))

# map ground truth to 2D
gt_map = np.zeros((8,4))
for i in range(len(R_mean)):
    gt_map[int(ch2xy[i,0]-1),int(ch2xy[i,1]-1)] = R_mean[i]

fig, ax =plt.subplots(sbs_mapped_arr.shape[1]+1,sbs_mapped_arr.shape[0], figsize=(20,12))

for i in range(sbs_mapped_arr.shape[0]):
    for j in range(sbs_mapped_arr.shape[1]+1):
        
        if j==0:
            # plot ground truth
            ax[j,i].imshow(gt_map, cmap='OrRd')
            
            if i<2:
                for q in range(sbs_save_query[i]):  
                    # plot queried coordinates as black dot
                    circle1 = plt.Circle((sbs_query_idx[q,1]-1, sbs_query_idx[q,0]-1), 0.1, color='k')
                    ax[j,i].add_patch(circle1)
                
            else:
                for q in np.linspace(sbs_save_query[i]-4,sbs_save_query[i]-1,4).astype(int):
                    # plot last 4 queried coordinates as black dot
                    circle1 = plt.Circle((sbs_query_idx[q,1]-1, sbs_query_idx[q,0]-1), 0.1, color='k')
                    ax[j,i].add_patch(circle1)      
            
                if i>=4:
                # plot best predicted coordinates as green dot (only starting from the 16th query)
                    circle2 = plt.Circle((sbs_best_query[int(sbs_save_query[i]-1),1]-1, sbs_best_query[int(sbs_save_query[i]-1),0]-1),
                                     0.2, color='g') 
                    ax[j,i].add_patch(circle2)
        else:
            if j==1:
                # plot estimated map                
                ax[j,i].imshow((sbs_mapped_arr[i,j-1]-np.min(sbs_mapped_arr[:,j-1]))/(np.max(sbs_mapped_arr[-1,j-1])-np.min(sbs_mapped_arr[:,j-1])),vmin =0, vmax =1, cmap='OrRd')

            elif j==2:
                # plot uncertainty
                ax[j,i].imshow((sbs_mapped_arr[i,j-1]-np.min(sbs_mapped_arr[:,j-1]))/(np.max(sbs_mapped_arr[:,j-1])-np.min(sbs_mapped_arr[:,j-1])),vmin =0, vmax =1, cmap='OrRd')
            elif j==3:
                # plot UCB map
                ax[j,i].imshow((sbs_mapped_arr[i,j-1]-np.min(sbs_mapped_arr[i,j-1]))/(np.max(sbs_mapped_arr[i,j-1])-np.min(sbs_mapped_arr[i,j-1])),vmin =0, vmax =1, cmap='OrRd')
            
         # remove tick 
        ax[j,i].set_xticklabels([])
        ax[j,i].set_xticks([])
        ax[j,i].set_yticklabels([])
        ax[j,i].set_yticks([])
        
        if j==0:
            ax[j,i].set_title(str(sbs_save_query[i]), fontsize=16)
        if j==0 and i==0:
            ax[j,i].set_ylabel('Ground truth', fontsize=16)
        if j==1 and i==0:
            ax[j,i].set_ylabel('Estimated map', fontsize=16)
        if j==2 and i==0:
            ax[j,i].set_ylabel('Uncertainty', fontsize=16)
        if j==3 and i==0:
            ax[j,i].set_ylabel('Acquisition function', fontsize=16)
            
# Save the plot to file (both picture and vector formats)        
plt.savefig(os.path.join(args.run1,'step_by_step_visualization.png'), bbox_inches='tight')
plt.savefig(os.path.join(args.run1,'step_by_step_visualization.svg'), format='svg')

# Clear plot
plt.clf()

# Log message to user
print('Hooray! Step by step visualization successfully generated!')
