import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFilePdf, faFileImage, faFileWord, faFileExcel, faFilePowerpoint, faUpload } from '@fortawesome/free-solid-svg-icons';
import Navbar from './components/Navbar';

interface Upload {
  filename: string;
  upload_date: string; // or Date, if you process it further
}

const Dashboard: React.FC = () => {
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [classificationResult, setClassificationResult] = useState<string | null>(null);
  const [textToClassify, setTextToClassify] = useState<string>('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get<{ uploads: Upload[] }>('http://localhost:5000/dashboard', {
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
    setLoading(true);
    const formData = new FormData();
    formData.append('file', acceptedFiles[0]);

    try {
      await axios.post<{ message: string }>('http://localhost:5000/upload', formData, {
        withCredentials: true,
      });

      setUploads((prevUploads) => [
        ...prevUploads,
        { filename: acceptedFiles[0].name, upload_date: new Date().toISOString() }, // Adjust the date format as needed
      ]);

      if (textToClassify) {
        const classifyResponse = await axios.post<{ category: string }>('http://localhost:5000/simple_classify', { text: textToClassify }, {
          withCredentials: true,
        });
        setClassificationResult(classifyResponse.data.category);
      }
    } catch (error) {
      console.error('Error uploading file or classifying text:', error);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf': return <FontAwesomeIcon icon={faFilePdf} size="3x" className="mb-2 text-red-500" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif': return <FontAwesomeIcon icon={faFileImage} size="3x" className="mb-2 text-blue-500" />;
      case 'doc':
      case 'docx': return <FontAwesomeIcon icon={faFileWord} size="3x" className="mb-2 text-blue-700" />;
      case 'xls':
      case 'xlsx': return <FontAwesomeIcon icon={faFileExcel} size="3x" className="mb-2 text-green-500" />;
      case 'ppt':
      case 'pptx': return <FontAwesomeIcon icon={faFilePowerpoint} size="3x" className="mb-2 text-orange-500" />;
      default: return <FontAwesomeIcon icon={faFilePdf} size="3x" className="mb-2 text-gray-500" />;
    }
  };

  const handleFileClick = (filename: string) => {
    window.open(`http://localhost:5000/uploads/${filename}`, '_blank');
  };

  const handleClassifyClick = async () => {
    try {
      const response = await axios.post<{ category: string }>('http://localhost:5000/simple_classify', { text: textToClassify }, {
        withCredentials: true,
      });
      setClassificationResult(response.data.category);
    } catch (error) {
      console.error('Error classifying text:', error);
    }
  };

  return (
    <>
      <Navbar />
      <div className="container mx-auto p-4 font-mono">
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

        {classificationResult && (
          <div className="mb-4 p-4 bg-green-100 text-green-800 rounded">
            <p>Classification Result: {classificationResult}</p>
          </div>
        )}

        <input
          type="text"
          placeholder="Enter text to classify"
          value={textToClassify}
          onChange={(e) => setTextToClassify(e.target.value)}
          className="border-2 border-gray-300 p-2 rounded mb-4 w-full"
        />
        <button
          onClick={handleClassifyClick}
          className="bg-blue-500 text-white p-2 rounded"
        >
          Classify Text
        </button>

        <h2 className="text-xl font-semibold mt-8 mb-4">Uploaded Files</h2>
        {loading && <p>Loading...</p>}
        <ul>
          {uploads.map((upload) => (
            <li key={upload.filename} className="mb-2">
              <button onClick={() => handleFileClick(upload.filename)} className="flex items-center">
                {getFileIcon(upload.filename)}
                <span className="ml-2">{upload.filename}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
};

export default Dashboard;
