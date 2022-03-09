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


#norm = clrs.Normalize(vmin=minimumValue, vmax=maximumValue) #linear norm


# density plot normalization
maximumValueRho = 1e-25
minimumValueRho = 1e-32

normRho = clrs.SymLogNorm(linthresh=minimumValueRho, vmin=minimumValueRho, vmax=maximumValueRho, base=10)




def createDataScipy(x, y, qty, bins=169, xyLim=100):
    x = np.concatenate((x, (-100, 100)))
    y = np.concatenate((y, (-100, 100)))
    qty = np.concatenate((qty, (0, 0)))

    rawBinnedData, xBins, yBins, _ = stats.binned_statistic_2d(x,
                                                               y,
                                                               qty,
                                                               bins=bins)

    binnedData = rawBinnedData
    binnedData[np.isnan(rawBinnedData)] = 0.0 # probably good enough for this pourpose

    #maskedBinnedData = ma.masked_array(rawBinnedData, np.isnan(rawBinnedData))
    #binnedData = maskedBinnedData.filled(maskedBinnedData.min() # this sollution may be more general

    binnedData = ndimage.gaussian_filter(binnedData, sigma=1.5)

    extent = [xBins[0], xBins[-1], yBins[0], yBins[-1]]

    return binnedData.T, extent



def makeSinglePlot(number, plotData, extent,
                    cmap='inferno', dpi=150,
                    norm=normRho):

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'$t={round(number * 0.01, 3):0.02f}$ Gyr', fontsize=16)


    # density plot ------- hardcoded cmap
    #ax[0].set_title('$t={:0.02f}$ gyr'.format(round(number * 0.01, 3)))
    ax.imshow(plotData, cmap='cubehelix', norm=norm,
     extent=extent, interpolation='nearest')
    ax.set_xlabel(r'x (kpc)')
    ax.set_ylabel(r'y (kpc)')
    #ax.set_xlim(extent[0], extent[1])
    #ax.set_ylim(extent[2], extent[3])

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    ax.set_aspect('equal')

    #Setting Colorbar
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='cubehelix'),
                 ax=ax, label=r'$\rho(\rm{cm} \rm{g}^{-3}$)',
                 shrink=1.0, aspect=20*1.4)



    fig.savefig(f'framesGas/snapshot_{number:04d}.jpg', dpi=dpi)
    plt.close(fig) # closing figure to reduce ram usage





if __name__ == '__main__':

    # hardcoded names for snapshots
    snapshotNames = [f'output/snapshot_{i:04d}.hdf5' for i in range(201)]

    for number, currentSnapshotName in enumerate(snapshotNames):

        # getting useful quantities from gas
        print(f'\033[F\033[F\033[F\033[FLoading snapshot {number:10.0f}')
        with h5py.File(currentSnapshotName, 'r') as file:
            gasPositions    = file['PartType0']['Coordinates'][:]
            gasId           = file['PartType0']['ParticleIDs'][:]
            gasMass         = file['PartType0']['Masses'][:]
            gasRho          = file['PartType0']['Density'][:] * rhoConversion / 10
            


        # saving useful quantities as a dict
        # TODO using a dataclass can be cleaner
        currentSnapshot =  {'pos'   : gasPositions,
                            'iord'  : gasId,
                            'mass'  : gasMass,
                            'rho'   : gasRho}



        # shifting (only gas particles)
        print(f'Shifting particles...')

        shiftMethod = 0



        if shiftMethod == 0:
            # into center of mass
            comPosition = np.array([np.sum(currentSnapshot['pos'][:, 0] * currentSnapshot['mass']),
                                    np.sum(currentSnapshot['pos'][:, 1] * currentSnapshot['mass']),
                                    np.sum(currentSnapshot['pos'][:, 2] * currentSnapshot['mass'])])

            #comPosition = comPosition / np.shape(currentSnapshot['pos'][:, 1])
            comPosition = comPosition / np.sum(currentSnapshot['mass'])

            currentSnapshot['pos'] = currentSnapshot['pos'] - comPosition

        elif shiftMethod == 1:
            # into highest density point
            peakRho = currentSnapshot['rho'].argmax()

            currentSnapshot['pos'] = currentSnapshot['pos'] - currentSnapshot['pos'][peakRho]



        edgeOn = True
        if edgeOn:
            # EDGE-ON
            # creating data slabs around |z| < 200 and |x|,|y| < 1000
            print(f'Creating data slab...')
            xyLim = 100


            # slab for density plot
            slabMaskRho = (currentSnapshot['pos'][:, 0] > -xyLim) & (currentSnapshot['pos'][:, 0] < xyLim) &\
                            (currentSnapshot['pos'][:, 1] > -xyLim) & (currentSnapshot['pos'][:, 1] < xyLim)

            # generating image data
            binnedDataRho, extentRho = createDataScipy(currentSnapshot['pos'][slabMaskRho][:, 0],
                                                 currentSnapshot['pos'][slabMaskRho][:, 1],
                                                 currentSnapshot['rho'][slabMaskRho][:],
                                                 xyLim=xyLim)
        else:
            # FACE-ON
            # creating data slabs around |x| < 200 and |y|,|z| < 50
            print(f'Creating data slab...')
            yzLim = 1000

            # slab for density plot
            slabMaskRho = (currentSnapshot['pos'][:, 1] > -yzLim) & (currentSnapshot['pos'][:, 1] < yzLim) &\
                            (currentSnapshot['pos'][:, 2] > -yzLim) & (currentSnapshot['pos'][:, 2] < yzLim)
    
            # generating image data
            binnedDataRho, extentRho = createDataScipy(currentSnapshot['pos'][slabMaskRho][:, 1],
                                                        currentSnapshot['pos'][slabMaskRho][:, 2],
                                                        currentSnapshot['rho'][slabMaskRho][:],
                                                        xyLim=yzLim)


        # plotting
        makeSinglePlot(number, binnedDataRho, extentRho)

        print(f'Finished plot {number:10.0f}')