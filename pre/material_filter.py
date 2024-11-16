import numpy as np
import h5py as h5


class crystal_system:
    # all filter function return an 0/1 array with size (n,) where n is the number of materials
    # then we can get intersection of several filter result to get what we want.
    # e.g. res1 is returned by filter for number of atoms (natom=3), res2 is returned by hexagonal filter
    # then we can use command like: (res1 & res2) to get what we request.

    def __init__(self, fname) -> None:
        self.fname = fname
        self.read_data_base()
    
    def read_data_base(self):
        fid = h5.File(self.fname,'r')
        self.cells = fid['cell'][()]
        self.atoms = fid['cell'][()]
        self.id = fid['id'][()]
        self.numbers_of_atoms = fid['numbers_of_atoms'][()]
        self.positions = fid['positions'][()]
        self.unique_ide = fid['unique_id'][()]
        fid.close()

    # def cell2abc(self):
    #     abc = np.zeros((self.cells[0],6)) # (n, 6) 6:a,b,c,alpha1, alpha2, alpha3
    #     for i in range()

    # todo: add more filters
    def filter_hexagonal(self):
        """
        cell: (n,3,3)
        """
        res = np.zeros(self.cells.shape[0])
        for n in range(len(self.cells)):
            if self.is_hexagonal_lattice(self.cells[n][0],self.cells[n][1],self.cells[n][2]):
                res[n] = 1
        
        return res.astype(int) # (n,) where 1 denotes True and 0 denotes False

    def atom_numbser_filter(self,natom=3):
        return (self.numbers_of_atoms == natom).astype(int)

    def abc_axis(self, abc_min=0, abc_max=20, axis='c'):
        if axis not in 'abc':
            raise Exception('specify axis:"a", "b" or "c" ')
        if axis == 'a': cell_length = np.linalg.norm(self.cells[:,0,:], axis=1)
        if axis == 'b': cell_length = np.linalg.norm(self.cells[:,1,:], axis=1)
        if axis == 'c': cell_length = np.linalg.norm(self.cells[:,2,:], axis=1)
        return np.where((cell_length<abc_max)&(abc_min<cell_length),1,0)
    
    
    def is_hexagonal_lattice(self,v1, v2, v3):
        # Calculate angles between vectors
        angle_v1_v2 = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angle_v1_v3 = np.degrees(np.arccos(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))))
        angle_v2_v3 = np.degrees(np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))))

        # Check if the angles are approximately equal to 60 degrees and 90 degrees
        return (
            (np.isclose(angle_v1_v2, 120, atol=1))and
            np.isclose(angle_v1_v3, 90, atol=1) and
            np.isclose(angle_v2_v3, 90, atol=1)
        )

if __name__ == '__main__':

    crystal = crystal_system('c2db-2022-11-30.h5')
    res_hex = crystal.filter_hexagonal()
    res_threeatoms = crystal.atom_numbser_filter()
    res_clengh = crystal.abc_axis(18,18.5,axis='c')
    res_alengh = crystal.abc_axis(3.25,3.75,axis='a')

    res3 = res_clengh & res_threeatoms & res_hex & res_alengh
    res3_index = list(np.where(res3==1)[0])

    a = np.linalg.norm(crystal.cells[res3_index,0], axis=1)
    b = np.linalg.norm(crystal.cells[res3_index,1], axis=1)
    c = np.linalg.norm(crystal.cells[res3_index,2], axis=1)





    
