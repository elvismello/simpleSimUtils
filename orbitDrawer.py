import numpy as np
import h5py
from matplotlib import pyplot as plt

# stylesheet from matplotlib
plt.style.use('./stylesheet.mplstyle')





if __name__ == '__main__':
    # hardcoded names for snapshots
    snapshotNames = [f'output/snapshot_{i:04d}.hdf5' for i in range(101)]


    comPositionList = []
    print('\n\n')
    for number, currentSnapshotName in enumerate(snapshotNames):

        # getting useful quantities from gas
        print(f'\033[F\033[FLoading snapshot {number:10.0f}')
        with h5py.File(currentSnapshotName, 'r') as file:
            gasPositions    = file['PartType2']['Coordinates'][:]
            gasId           = file['PartType2']['ParticleIDs'][:]
            #gasMass         = file['PartType0']['Masses'][:]
        
        # saving useful quantities as a dict
        # TODO using a dataclass can be cleaner
        currentSnapshot =  {'pos'   : gasPositions,
                            'iord'  : gasId}
                            #'mass'  : gasMass}



        print('Finding COM...')

        relevantParticles = currentSnapshot

        comPosition = np.array([np.sum(relevantParticles['pos'][:, 0]),
                                np.sum(relevantParticles['pos'][:, 1]),
                                np.sum(relevantParticles['pos'][:, 2])])

        comPosition = comPosition / np.shape(currentSnapshot['pos'][:, 1])

        #relevantParticles = {'pos':currentSnapshot['pos'][currentSnapshot['iord'] > 3000000],
        #                    'iord':currentSnapshot['iord'][currentSnapshot['iord'] > 3000000],
        #                    'mass':currentSnapshot['mass'][currentSnapshot['iord'] > 3000000]} 

        # into center of mass
        #comPosition = np.array([np.sum(relevantParticles['pos'][:, 0] * relevantParticles['mass']),
        #                        np.sum(relevantParticles['pos'][:, 1] * relevantParticles['mass']),
        #                        np.sum(relevantParticles['pos'][:, 2] * relevantParticles['mass'])])

        #comPosition = comPosition / np.sum(relevantParticles['mass'])

        comPositionList.append(comPosition)

    comPositionList = np.array(comPositionList)

    print('Making plot...')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(comPositionList[:, 0], comPositionList[:, 1])


    fig.savefig('diskOrbitXY.jpg', dpi=300)



