//! Array flags implementation
//!
//! This module provides array flags functionality,
//! equivalent to NumPy's flagsobject.c

use bitflags::bitflags;

bitflags! {
    /// Array flags
    ///
    /// These flags control various properties and behaviors of arrays,
    /// equivalent to NumPy's array flags.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ArrayFlags: u32 {
        /// Array data is C-style contiguous
        const C_CONTIGUOUS = 0x0001;
        /// Array data is Fortran-style contiguous
        const F_CONTIGUOUS = 0x0002;
        /// Array owns its data
        const OWNDATA = 0x0004;
        /// Array is writeable
        const WRITEABLE = 0x0008;
        /// Array is aligned
        const ALIGNED = 0x0010;
        /// Array is write-back-if-copy
        const WRITEBACKIFCOPY = 0x0020;
        /// Array update-if-copy (deprecated)
        const UPDATEIFCOPY = 0x0040;
    }
}

impl Default for ArrayFlags {
    fn default() -> Self {
        ArrayFlags::C_CONTIGUOUS | ArrayFlags::WRITEABLE | ArrayFlags::ALIGNED | ArrayFlags::OWNDATA
    }
}

