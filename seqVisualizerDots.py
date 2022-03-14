import stat
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cm
import numpy as np
from scipy import stats, ndimage


# stylesheet from matplotlib
plt.style.use('./stylesheet.mplstyle')


# constants
kB = 1.38064852e-23 # [m**2 kg s**-2 K**-1]
mH = 1.6726219e-27  # [kg]
mu = 0.6            # fully ionized gas assuming primordial abundance with X=0.76
jouleTokeV = 6.242e15
mSolarTog = 1.989e33# [g]
pcTocm = 3.086e18   # [cm]
rhoConversion = mSolarTog / (pcTocm ** 3) # [g * cm**-3]

edgeOn = False



#norm = clrs.Normalize(vmin=minimumValue, vmax=maximumValue) #linear norm



def makeDotsPlot(number, plotData, xyLim, dpi=150):

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('$t={:0.02f}$ Gyr'.format(round(number * 0.025, 3)), fontsize=16)


    # density plot ------- hardcoded cmap
    #ax[0].set_title('$t={:0.02f}$ gyr'.format(round(number * 0.01, 3)))
    ax.scatter(plotData[0], plotData[1], s=0.1, c='white', marker='.')
    ax.set_xlabel(r'x (kpc)')
    ax.set_ylabel(r'y (kpc)')
    ax.set_xlim(-xyLim, xyLim)
    ax.set_ylim(-xyLim, xyLim)
    ax.set_aspect('equal')
    ax.set(facecolor='black')


    fig.savefig(f'framesDisk/snapshot_{number:04d}.jpg', dpi=dpi)
    plt.close(fig) # closing figure to reduce ram usage





if __name__ == '__main__':

    # hardcoded names for snapshots
    snapshotNames = [f'output/snapshot_{i:04d}.hdf5' for i in range(101)]

    print('\n\n\n')
    for number, currentSnapshotName in enumerate(snapshotNames):

        # getting useful quantities from gas
        print(f'\033[F\033[F\033[F\033[FLoading snapshot {number:10.0f}')
        with h5py.File(currentSnapshotName, 'r') as file:
            gasPositions    = file['PartType2']['Coordinates'][:]
            gasId           = file['PartType2']['ParticleIDs'][:]
            gasMass = np.zeros(gasPositions.shape[0])
            gasMass.fill(file['Header'].attrs['MassTable'][2])


        # saving useful quantities as a dict
        # TODO using a dataclass can be cleaner
        currentSnapshot =  {'pos': gasPositions,
                            'iord': gasId,
                            'mass': gasMass}



        # shifting (only gas particles)
        print(f'Shifting particles...')

        #used to select particles, if needed
        relevantParticles = currentSnapshot

        # into center of mass
        comPosition = np.array([np.sum(relevantParticles['pos'][:, 0]),
                                np.sum(relevantParticles['pos'][:, 1]),
                                np.sum(relevantParticles['pos'][:, 2])])

        comPosition = comPosition / np.shape(currentSnapshot['pos'][:, 1])

        currentSnapshot['pos'] = currentSnapshot['pos'] - comPosition




        if edgeOn:
            # EDGE-ON
            # creating data slabs around |z| < 200 and |x|,|y| < 1000
            print(f'Creating data slab...')
            xyLim = 50


            # slab for density plot
            slabMaskRho = (currentSnapshot['pos'][:, 0] > -xyLim) & (currentSnapshot['pos'][:, 0] < xyLim) &\
                            (currentSnapshot['pos'][:, 1] > -xyLim) & (currentSnapshot['pos'][:, 1] < xyLim)

            # generating image data

            xPos = currentSnapshot['pos'][slabMaskRho][:, 0]
            yPos = currentSnapshot['pos'][slabMaskRho][:, 1]
            zPos = currentSnapshot['pos'][slabMaskRho][:, 2]

            # plotting
            makeDotsPlot(number, [xPos, yPos], xyLim)


        else:
            # FACE-ON
            # creating data slabs around |x| < 200 and |y|,|z| < 50
            print(f'Creating data slab...')
            yzLim = 55

            # slab for density plot
            slabMaskRho = (currentSnapshot['pos'][:, 1] > -yzLim) & (currentSnapshot['pos'][:, 1] < yzLim) &\
                            (currentSnapshot['pos'][:, 2] > -yzLim) & (currentSnapshot['pos'][:, 2] < yzLim)
    
            # generating image data
            xPos = currentSnapshot['pos'][slabMaskRho][:, 0]
            yPos = currentSnapshot['pos'][slabMaskRho][:, 1]
            zPos = currentSnapshot['pos'][slabMaskRho][:, 2]

            # plotting
            makeDotsPlot(number, [yPos, zPos], yzLim)

        

        print(f'Finished plot {number:10.0f}')