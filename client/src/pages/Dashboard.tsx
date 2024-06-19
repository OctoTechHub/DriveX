import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFilePdf, faFileImage, faFileWord, faFileExcel, faFilePowerpoint, faUpload } from '@fortawesome/free-solid-svg-icons';
import Navbar from './components/Navbar';

const Dashboard: React.FC = () => {
  const [uploads, setUploads] = useState<any[]>([]);
  const [loading, setLoading] = useState(false); // Loading state
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/dashboard', {
          withCredentials: true,
        });
        setUploads(response.data.uploads);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        navigate('/signin');
      }
    };

    fetchData();
  }, [navigate]);

  const onDrop = async (acceptedFiles: File[]) => {
    setLoading(true); // Set loading to true when starting upload
    const formData = new FormData();
    formData.append('file', acceptedFiles[0]);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        withCredentials: true,
      });
      console.log(response.data);
      setUploads([...uploads, response.data.upload]);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false); // Set loading to false after upload completes (success or error)
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    if (extension === 'pdf') {
      return <FontAwesomeIcon icon={faFilePdf} size="3x" className="mb-2 text-red-500" />;
    } else if (['jpg', 'jpeg', 'png', 'gif'].includes(extension as string)) {
      return <FontAwesomeIcon icon={faFileImage} size="3x" className="mb-2 text-blue-500" />;
    } else if (['doc', 'docx'].includes(extension as string)) {
      return <FontAwesomeIcon icon={faFileWord} size="3x" className="mb-2 text-blue-700" />;
    } else if (['xls', 'xlsx'].includes(extension as string)) {
      return <FontAwesomeIcon icon={faFileExcel} size="3x" className="mb-2 text-green-500" />;
    } else if (['ppt', 'pptx'].includes(extension as string)) {
      return <FontAwesomeIcon icon={faFilePowerpoint} size="3x" className="mb-2 text-orange-500" />;
    } else {
      return <FontAwesomeIcon icon={faFilePdf} size="3x" className="mb-2 text-gray-500" />;
    }
  };

  const handleFileClick = (filename: string) => {
    window.open(`http://localhost:5000/uploads/${filename}`, '_blank');
  };

  return (
    <>
      <Navbar />
      <div className="container mx-auto p-4">
        <h1 className="text-2xl font-bold mb-4">Welcome Back, {localStorage.getItem('username')}!</h1>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed p-6 rounded-lg mb-4 cursor-pointer text-center ${
            isDragActive ? 'border-blue-500' : 'border-gray-300'
          }`}
        >
          <input {...getInputProps()} />
          <FontAwesomeIcon icon={faUpload} size="3x" className="mb-2 text-gray-400" />
          {isDragActive ? (
            <p className="text-blue-500">Drop the files here ...</p>
          ) : (
            <p>Drag 'n' drop some files here, or click to select files</p>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {uploads.map((upload, index) => (
            <div key={index} className="bg-white shadow-md rounded-lg p-4 cursor-pointer" onClick={() => handleFileClick(upload.filename)}>
              <div className="flex items-center justify-center">
                {getFileIcon(upload.filename)}
              </div>
              <p className="text-lg font-semibold text-center">{upload.filename}</p>
              <p className="text-sm text-gray-500 text-center">{new Date(upload.upload_date).toLocaleString()}</p>
            </div>
          ))}
        </div>

        {loading && (
          <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-gray-800 bg-opacity-75 text-white z-50">
            <p>Uploading...</p>
          </div>
        )}
      </div>
    </>
  );
};

export default Dashboard;
