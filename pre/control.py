import numpy as np
from job_control import MaterialJobController
from mpi4py import MPI
import h5py as h5
import sys
from material_filter import crystal_system
from gpaw import GPAW, PW, FermiDirac
import os
import random
import shutil
from ase.parallel import paropen, world,parprint


# This program will read a list of material index and then calculate scf and write h5 wfc

class control:
    def __init__(self, library_path: str, material_index: list = []) -> None:
        self.library_path = library_path
        self.material_index = material_index
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

    def runscf(self, ecutoff_dft=100, gwcutoff=80):
        for index in self.material_index:
            parprint('scf calculation for wfs-%s' % index)
            self.controller = MaterialJobController(library_path=self.library_path,
                                                    material_idx=index,
                                                    ecutoff_dft=ecutoff_dft,
                                                    ecutoff_gw=gwcutoff)
            self.controller.scf()
            self._writewfc(controller=self.controller, index=index)


    def runband(self, ecutoff_dft=100, nkpt=60, kpath='GMKG'):
        for index in self.material_index:
            parprint('scf calculation for wfs-%s' % index);sys.stdout.flush()
            self.controller = MaterialJobController(library_path=self.library_path,
                                                    material_idx=index,
                                                    ecutoff_dft=ecutoff_dft,
                                                    ecutoff_gw=80)
            self.controller.scf()
            nelectron = self.controller.atoms.calc.get_number_of_electrons()
            self.controller.bands(nbands=int(nelectron),
                     kpath={'path':kpath, 'npoints': nkpt},
                     convergece_band=8)
            bd = self.controller.calc_band.band_structure()
            self.controller.atoms.calc = self.controller.calc_band

            self._writewfc_band(controller=self.controller, index=index)


    def rungw(self,  ecutoff_dft=100, gwcutoff=80, gwrange = [-1,1]):
        """
        :param gwrange is range of gw band, 0 is VBM. therefore [-1,1] is VBM-1 to VBM, last index is not included
        """
        for index in self.material_index:
            if self.rank == 0: print('gw calculation for wfs-%s'%index); sys.stdout.flush()
            self.controller = MaterialJobController(library_path=self.library_path,
                                                    material_idx=index,
                                                    ecutoff_dft=ecutoff_dft,
                                                    ecutoff_gw=gwcutoff)

            if not os.path.exists(self.controller.unique_id+'-scf.gpw'):
                raise Exception('%s-scf.gpw is not exsisting' % self.controller.unique_id)
            self.controller.atoms.calc = GPAW(self.controller.unique_id+'-scf.gpw')

            nele = int(self.controller.atoms.calc.get_number_of_electrons())
            if self.rank == 0: print('band_range', nele // 2 + gwrange[0], nele // 2 + gwrange[1]); sys.stdout.flush()
            self.controller.g0w0(bands_range=(nele//2+gwrange[0], nele//2+gwrange[1]))
            self._writegw(controller=self.controller, index=index, gwrange=gwrange)


    def _writewfc(self, controller, index):
        output_dir='figures_try_effect'
        cell = np.array(self.controller.atoms.cell)
        nbnd = controller.atoms.calc.get_number_of_bands()  # int
        nelectron = int(controller.atoms.calc.get_number_of_electrons())  # int
        ibzk = controller.atoms.calc.get_ibz_k_points()  # array: [nk, 3]
        grid = controller.atoms.calc.get_number_of_grid_points()
        fermilevel = controller.atoms.calc.get_fermi_level()
        nsuper = 3
        parprint(
            'Controller speaking:\n  nelectron: %s  \n  nbnd: %s  \n  ibzk: %s  \n  grid:  %s  \n  fermilevel:%.4f' % (
            nelectron, nbnd, ibzk, grid, fermilevel))
        if self.rank == 0:
            file_path=os.path.join(output_dir,'wfs-%s.hdf5' % index)
            f = h5.File(file_path, 'w'); sys.stdout.flush()
            f.create_dataset('data', (len(ibzk), nelectron, grid[0], grid[1], grid[2]), dtype=complex)
            f.create_dataset('charge_density', (len(ibzk), nelectron, grid[0], grid[1], grid[2]))
            f.create_dataset('energy', (len(ibzk), nelectron))
            f.create_dataset('GWenergy', (len(ibzk), nelectron))
            f['GWenergy'][()] = np.ones((len(ibzk), nelectron)) * (-1000)
            f.create_dataset('kpt', (len(ibzk), nelectron, 3))
            f.create_dataset('cell', (len(ibzk), nelectron, 3, 3))

            # 11/19/2023 update superbands
            f.create_dataset('super_band', (
            len(ibzk), nelectron, nsuper, grid[0], grid[1], grid[2]))  # here we calculate 3 super bands in total
            # ---------------------------

        # 11/19/2023 update superbands
        # print('creating super state...')
        self.super_state = np.zeros((nsuper, grid[0], grid[1], grid[2]))
        for ik in range(len(ibzk)):
            for ib in range(nbnd):
                iblock = self.ibnd2super(nsuper=nsuper, nbnd=nbnd, band_index=ib)
                self.super_state[iblock] = self.super_state[iblock] + np.abs(
                    self.controller.atoms.calc.get_pseudo_wave_function(band=ib, kpt=ik))
        # print('super state created!')
        # 11/19/2023 update superbands


        # 01/22/2024 super_state_occupied
        self.super_state[0] = np.zeros_like(self.super_state[0])
        for ik in range(len(ibzk)):
            for ib in range(nelectron//2):
                self.super_state[0] = self.super_state[0] + np.abs(
                    self.controller.atoms.calc.get_pseudo_wave_function(band=ib, kpt=ik))
        # 01/22/2024 super_state_occupied
                

        temp_charge_density = self.controller.atoms.calc.get_pseudo_density()
        for ik in range(len(ibzk)):
            parprint('  - writting wfs-%s:' % index, ik, ' kpt')
            sys.stdout.flush()
            temp_eigenvalues = self.controller.atoms.calc.get_eigenvalues(kpt=ik)
            for ivb in range(nelectron):  # only valence states are written
                temp_wfs = controller.atoms.calc.get_pseudo_wave_function(band=ivb, kpt=ik)

                if self.rank == 0:
                    # print('    --wfs.shape:',temp_wfs.shape)
                    f['cell'][ik, ivb] = cell
                    f['kpt'][ik, ivb] = ibzk[ik]
                    f['charge_density'][ik, ivb] = temp_charge_density
                    f['data'][ik, ivb] = temp_wfs
                    f['energy'][ik, ivb] = temp_eigenvalues[ivb] - fermilevel
                    f['super_band'][ik, ivb] = self.super_state

                self.comm.barrier()


    def _writewfc_band(self, controller, index):

        cell = np.array(self.controller.atoms.cell)
        nbnd = controller.atoms.calc.get_number_of_bands()  # int
        nelectron = int(controller.atoms.calc.get_number_of_electrons())  # int
        ibzk = controller.atoms.calc.get_ibz_k_points()  # array: [nk, 3]
        grid = controller.atoms.calc.get_number_of_grid_points()
        fermilevel = controller.atoms.calc.get_fermi_level()
        nsuper = 3
        print('Controller speaking:\n  nelectron: %s  \n  nbnd: %s  \n  ibzk: %s  \n  grid:  %s  \n  fermilevel:%.4f' % (nelectron, nbnd, ibzk, grid, fermilevel))

        if self.rank == 0:
            f = h5.File('wfs-%s.hdf5' % index, 'w')
            f.create_dataset('data', (len(ibzk), 2, grid[0], grid[1], grid[2]), dtype=complex)
            f.create_dataset('charge_density', (len(ibzk), 2, grid[0], grid[1], grid[2]))
            f.create_dataset('energy', (len(ibzk), 2))
            f.create_dataset('GWenergy', (len(ibzk), 2))
            f['GWenergy'][()] = np.ones((len(ibzk), 2)) * (-1000)
            f.create_dataset('kpt', (len(ibzk), 2, 3))
            f.create_dataset('cell', (len(ibzk), 2, 3, 3))

            # 11/19/2023 update superbands
            f.create_dataset('super_band', (len(ibzk), 2, nsuper ,grid[0], grid[1], grid[2])) # here we calculate 3 super bands in total
            #---------------------------

        # 11/19/2023 update superbands
        print('creating super state...')
        self.super_state = np.zeros((nsuper, grid[0], grid[1], grid[2]))
        for ik in range(len(ibzk)):
            for ib in range(nbnd):
                iblock = self.ibnd2super(nsuper=nsuper, nbnd=nbnd, band_index=ib)
                self.super_state[iblock] = self.super_state[iblock] + np.abs(self.controller.atoms.calc.get_pseudo_wave_function(band=ib, kpt=ik))
        print('super state created!')
        # 11/19/2023 update superbands


        # 01/22/2024 super_state_occupied
        self.super_state[0] = np.zeros_like(self.super_state[0])
        for ik in range(len(ibzk)):
            for ib in range(nelectron//2):
                self.super_state[0] = self.super_state[0] + np.abs(
                    self.controller.atoms.calc.get_pseudo_wave_function(band=ib, kpt=ik))
        # 01/22/2024 super_state_occupied

        temp_charge_density = self.controller.atoms.calc.get_pseudo_density()
        for ik in range(len(ibzk)):
            print('  - writting wfs-%s:' % index, ik, ' kpt');
            sys.stdout.flush()
            temp_eigenvalues = self.controller.atoms.calc.get_eigenvalues(kpt=ik)
            for ivb in range(nelectron//2-1,nelectron//2+1):  # only valence states are written
                temp_wfs = controller.atoms.calc.get_pseudo_wave_function(band=ivb, kpt=ik)

                if self.rank == 0:
                    # print('    --wfs.shape:',temp_wfs.shape)
                    f['cell'][ik, ivb - (nelectron//2-1)] = cell
                    f['kpt'][ik, ivb - (nelectron//2-1)] = ibzk[ik]
                    f['charge_density'][ik, ivb - (nelectron//2-1)] = temp_charge_density
                    f['data'][ik, ivb - (nelectron//2-1)] = temp_wfs
                    f['energy'][ik, ivb - (nelectron//2-1)] = temp_eigenvalues[ivb] - fermilevel
                    f['super_band'][ik, ivb - (nelectron//2-1)] = self.super_state

                self.comm.barrier()

    def _writegw(self, controller, index, gwrange):

        cell = np.array(self.controller.atoms.cell)
        nbnd = controller.atoms.calc.get_number_of_bands()  # int
        nelectron = int(controller.atoms.calc.get_number_of_electrons())  # int
        ibzk = controller.atoms.calc.get_ibz_k_points()  # array: [nk, 3]
        grid = controller.atoms.calc.get_number_of_grid_points()
        fermilevel = controller.atoms.calc.get_fermi_level()
        # print(
        #     'Controller speaking:\n  nelectron: %s  \n  nbnd: %s  \n  ibzk: %s  \n  grid:  %s  \n  fermilevel:%.4f' % (
        #     nelectron, nbnd, ibzk, grid, fermilevel))

        if not os.path.exists('wfs-%s.hdf5' % index):
            raise Exception('wfs is missing %s'% index)

        if self.rank == 0:
            f = h5.File('wfs-%s.hdf5' % index, 'a')

        for ik in range(len(ibzk)):
            if self.rank == 0: print('  - writting wfs-%s:' % index, ik, ' kpt');sys.stdout.flush()
            for ivb in range(nelectron):  # only valence states are written
                if self.rank == 0:
                    # print('    --wfs.shape:',temp_wfs.shape)
                    if nelectron//2 +gwrange[0] <= ivb < nelectron//2 +gwrange[1]: # todo:
                        f['GWenergy'][ik, ivb] = controller.result['qp'][0,ik, ivb-(nelectron//2 +gwrange[0])] - fermilevel
                self.comm.barrier()

        if self.rank == 0:
            f.close()
            # todo: write eigenvalue and structure data

        self.comm.barrier()

    def ibnd2super(self, nbnd, band_index, nsuper=3):
        """
        return: block index of a band index
        """
        nband_block = nbnd // nsuper
        return band_index // (nband_block + 1)


if __name__ == "__main__":
    crystal = crystal_system('c2db-2022-11-30.h5')
    '''
    # find materials index using crystal class
    res_hex = crystal.filter_hexagonal()
    res_threeatoms = crystal.atom_numbser_filter(3)
    res_clengh = crystal.abc_axis(18, 18.5, axis='c')
    res_alengh = crystal.abc_axis(3.25, 3.75, axis='a')
    # intersection of all filters:
    res3 = res_clengh & res_threeatoms & res_hex & res_alengh
    materials_index = list(np.where(res3 == 1)[0])
    '''
    number_range=[i for i in range(len(crystal.numbers_of_atoms)) if i != 6161]
    materials_index = random.sample(number_range, 3)

    # Run SCF
    if world.rank==0:
        output_dirs=['figures_try_effect','scf_gpw']
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)  # 删除整个output_dir文件夹及其内容
            os.makedirs(output_dir)  # 重新创建空的output_dir文件夹
    
    run = 'scf'
    con = control(material_index=materials_index,
                  library_path='c2db-2022-11-30.h5')
    if run == 'scf':
        con.runscf(ecutoff_dft=100, gwcutoff=80)  # turn off this once you finish dft!!!!
    if run == 'band':
        con.runband(ecutoff_dft=400, nkpt=80, kpath='GMKG') # turn off this once you finish dft!!!!
    if run == 'gw':
        con.rungw(ecutoff_dft=400, gwcutoff=80, gwrange=[-1, 1])
