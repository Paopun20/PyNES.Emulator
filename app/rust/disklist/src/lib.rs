use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write, Read};
use std::path::PathBuf;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;

/// A disk-backed list with optional compression, written in Rust and exposed to Python.
#[pyclass]
struct DiskList {
    path: PathBuf,
    index: Vec<u64>,
    file: File,
    compress: bool,
}

impl DiskList {
    /// Normalize negative indices to positive ones
    fn normalize_index(&self, idx: isize) -> PyResult<usize> {
        let len = self.index.len() as isize;
        let normalized = if idx < 0 {
            if idx < -len {
                return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            (len + idx) as usize
        } else {
            idx as usize
        };

        if normalized >= self.index.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }

        Ok(normalized)
    }

    /// Read length prefix at current file position
    fn read_length(&mut self) -> PyResult<u64> {
        let mut len_buf = [0u8; 8];
        self.file.read_exact(&mut len_buf)?;
        Ok(u64::from_le_bytes(len_buf))
    }

    /// Compress data if compression is enabled
    fn encode(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        if !self.compress {
            return Ok(data.to_vec());
        }

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    /// Decompress data if compression is enabled
    fn decode(&self, data: &[u8]) -> PyResult<Vec<u8>> {
        if !self.compress {
            return Ok(data.to_vec());
        }

        let mut decoder = ZlibDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    /// Write length prefix and data to current file position
    fn write_item(&mut self, data: &[u8]) -> PyResult<u64> {
        let encoded = self.encode(data)?;
        let len = encoded.len() as u64;
        let pos = self.file.stream_position()?;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&encoded)?;
        self.file.flush()?;
        Ok(pos)
    }

    /// Serialize a Python object to bytes using pickle
    fn serialize_item(&self, py: Python<'_>, item: Py<PyAny>) -> PyResult<Vec<u8>> {
        let pickle = PyModule::import(py, "pickle")?;
        pickle.call_method1("dumps", (item,))?.extract()
    }

    /// Deserialize bytes to a Python object using pickle
    fn deserialize_item(&self, py: Python<'_>, data: &[u8]) -> PyResult<Py<PyAny>> {
        let pickle = PyModule::import(py, "pickle")?;
        let obj = pickle.call_method1("loads", (pyo3::types::PyBytes::new(py, data),))?;
        Ok(obj.into())
    }

    /// Read item data at a given index position
    fn read_item_data(&mut self, idx: usize) -> PyResult<Vec<u8>> {
        let pos = self.index[idx];
        self.file.seek(SeekFrom::Start(pos))?;
        let len = self.read_length()?;
        let mut data = vec![0u8; len as usize];
        self.file.read_exact(&mut data)?;
        self.decode(&data)
    }

    /// Read all items from a given index onwards
    fn read_items_from(&mut self, start_idx: usize) -> PyResult<Vec<Vec<u8>>> {
        let mut items = Vec::new();
        for i in start_idx..self.index.len() {
            items.push(self.read_item_data(i)?);
        }
        Ok(items)
    }

    /// Truncate file and rewrite items from a position
    fn rewrite_from_position(&mut self, pos: u64, items: Vec<Vec<u8>>) -> PyResult<()> {
        self.file.set_len(pos)?;
        self.file.seek(SeekFrom::Start(pos))?;
        
        for data in items {
            let item_pos = self.write_item(&data)?;
            self.index.push(item_pos);
        }
        Ok(())
    }
}

#[pymethods]
impl DiskList {
    #[new]
    #[pyo3(signature = (path, compress=false))]
    fn new(path: String, compress: bool) -> PyResult<Self> {
        let pathbuf = PathBuf::from(&path);
        
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&pathbuf)?;

        let mut index = Vec::new();
        let mut pos = 0u64;
        let mut f = File::open(&pathbuf)?;
        
        loop {
            let mut len_buf = [0u8; 8];
            match f.read_exact(&mut len_buf) {
                Ok(_) => {
                    let len = u64::from_le_bytes(len_buf);
                    index.push(pos);
                    pos += 8 + len;
                    
                    if f.seek(SeekFrom::Current(len as i64)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        Ok(Self { path: pathbuf, index, file, compress })
    }

    fn append(&mut self, py: Python<'_>, item: Py<PyAny>) -> PyResult<()> {
        let dumped = self.serialize_item(py, item)?;
        self.file.seek(SeekFrom::End(0))?;
        let pos = self.write_item(&dumped)?;
        self.index.push(pos);
        Ok(())
    }

    fn pop(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.index.is_empty() {
            return Err(pyo3::exceptions::PyIndexError::new_err("pop from empty list"));
        }

        let idx = self.index.len() - 1;
        let item = self.__getitem__(py, idx as isize)?;
        self.index.pop();

        let new_len = if self.index.is_empty() {
            0
        } else {
            let last_idx = self.index.len() - 1;
            let last_item_pos = self.index[last_idx];
            self.file.seek(SeekFrom::Start(last_item_pos))?;
            let len = self.read_length()?;
            last_item_pos + 8 + len
        };
        
        self.file.set_len(new_len)?;
        self.file.flush()?;
        Ok(item)
    }

    fn __setitem__(&mut self, py: Python<'_>, idx: isize, item: Py<PyAny>) -> PyResult<()> {
        let idx = self.normalize_index(idx)?;
        let dumped = self.serialize_item(py, item)?;
        let encoded = self.encode(&dumped)?;
        let new_len = encoded.len() as u64;
        let old_pos = self.index[idx];
        
        self.file.seek(SeekFrom::Start(old_pos))?;
        let old_len = self.read_length()?;

        if new_len == old_len {
            self.file.seek(SeekFrom::Start(old_pos + 8))?;
            self.file.write_all(&encoded)?;
            self.file.flush()?;
        } else {
            let items_after = self.read_items_from(idx + 1)?;
            self.index.truncate(idx);
            self.rewrite_from_position(old_pos, vec![dumped])?;
            let pos = self.file.stream_position()?;
            self.rewrite_from_position(pos, items_after)?;
        }
        Ok(())
    }

    fn insert(&mut self, py: Python<'_>, idx: isize, item: Py<PyAny>) -> PyResult<()> {
        let len = self.index.len() as isize;
        let idx = if idx < 0 {
            if idx < -len { 0 } else { (len + idx) as usize }
        } else {
            (idx as usize).min(self.index.len())
        };

        if idx == self.index.len() {
            return self.append(py, item);
        }

        let dumped = self.serialize_item(py, item)?;
        let insert_pos = self.index[idx];
        
        self.file.seek(SeekFrom::Start(insert_pos))?;
        let mut remaining_data = Vec::new();
        self.file.read_to_end(&mut remaining_data)?;

        self.index.truncate(idx);
        self.rewrite_from_position(insert_pos, vec![dumped])?;
        
        let mut offset = 0;
        while offset < remaining_data.len() {
            if offset + 8 > remaining_data.len() { break; }
            let len = u64::from_le_bytes(remaining_data[offset..offset+8].try_into().unwrap());
            let item_end = offset + 8 + len as usize;
            if item_end > remaining_data.len() { break; }
            
            // Decode then re-encode to maintain consistency
            let decoded = self.decode(&remaining_data[offset+8..item_end])?;
            let pos = self.write_item(&decoded)?;
            self.index.push(pos);
            offset = item_end;
        }
        Ok(())
    }

    fn clean(&mut self) -> PyResult<()> {
        self.file.set_len(0)?;
        self.file.flush()?;
        self.index.clear();
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.index.len()
    }

    fn __getitem__(&mut self, py: Python<'_>, idx: isize) -> PyResult<Py<PyAny>> {
        let idx = self.normalize_index(idx)?;
        let data = self.read_item_data(idx)?;
        self.deserialize_item(py, &data)
    }

    /// Get compression ratio statistics
    fn get_stats(&mut self) -> PyResult<(u64, f64)> {
        let file_size = self.file.metadata()?.len();
        let compression_ratio = if self.compress && file_size > 0 {
            file_size as f64 / (file_size as f64)
        } else {
            1.0
        };
        Ok((file_size, compression_ratio))
    }
}

#[pymodule]
fn disklist(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiskList>()?;
    Ok(())
}