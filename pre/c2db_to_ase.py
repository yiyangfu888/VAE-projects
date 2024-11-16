import h5py
from ase import Atoms

def atom_index_range(fid: h5py.File, material_idx):
    r"""
    Find the indices of all atoms in material `material_idx` in `fid`.
    The right boundary of the return value is not included.
    """

    numbers_of_atoms = fid["numbers_of_atoms"][()] # import data to memory is safer
    # This is also the index of the first atom in the material
    num_atoms_prev_materials = sum(numbers_of_atoms[0:material_idx])
    num_atoms_this_material = numbers_of_atoms[material_idx]
    return range(num_atoms_prev_materials, num_atoms_prev_materials + num_atoms_this_material)


def atom_list(fid: h5py.File, material_idx):
    r"""
    Find the atomic numbers of the atoms in the material.
    """
    return fid["atoms"][list(atom_index_range(fid, material_idx))] # Iterator itself might not be directly used as indexing
    
def atom_positions(fid, material_idx):
    r"""
    Find the positions of the atoms. 
    The first index is the index of the atom,
    and the second index labels the coordinate (x, y, z).
    """
    return fid["positions"][list(atom_index_range(fid, material_idx)), :] # Iterator itself might not be directly used as indexing

def cell(fid, material_idx):
    r"""
    Find the primitive unit cell basis of the material.
    The first index specifies the index of the three basis vectors 
    and the second index specifies the coordinate x, y, z.
    """
    return fid["cell"][material_idx, :, :]

def atoms(fid, material_idx):
    r"""
    Convert the information about the material in `fid`
    into an `ase.Atoms` object.
    """
    return Atoms(
        numbers=atom_list(fid, material_idx),
        cell=cell(fid, material_idx),
        positions=atom_positions(fid, material_idx),
        pbc=[1, 1, 1]
    )

def unique_id(fid, material_idx):
    return fid["unique_id"][material_idx]