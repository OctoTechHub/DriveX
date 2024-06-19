import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload } from '@fortawesome/free-solid-svg-icons';
import Navbar from './components/Navbar';

const Dashboard: React.FC = () => {
  const [uploads, setUploads] = useState<any[]>([]);
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
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <>
    <Navbar/>
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

      <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
        <h2 className="text-xl font-semibold mb-4">Uploads:</h2>
        <ul className="list-disc pl-5">
          {uploads.map((upload, index) => (
            <li key={index} className="mb-2">
              <a
                href={`http://localhost:5000/uploads/${upload.filename}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                {upload.filename}
              </a> - {new Date(upload.upload_date).toLocaleString()}
            </li>
          ))}
        </ul>
      </div>
    </div>
    </>
  );
};

export default Dashboard;
