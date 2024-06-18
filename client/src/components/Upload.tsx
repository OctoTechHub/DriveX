import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { ReactElement, JSXElementConstructor, ReactNode, ReactPortal, Key } from 'react';

const Upload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploads, setUploads] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/dashboard', {
          params: {
            username: localStorage.getItem('username'),
            password: localStorage.getItem('password'),
          },
        });
        setUploads(response.data.uploads);
      } catch (error) {
        console.error(error);
        navigate('/signin');
      }
    };

    fetchData();
  }, [navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) return; 

    const formData = new FormData();
    formData.append('file', file);
    const username = localStorage.getItem('username');
    const password = localStorage.getItem('password');
    if (username && password) {
      formData.append('username', username);
      formData.append('password', password);
    } else {
      console.error('Username or password is not set');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);
      console.log(response.data);
      setUploads([...uploads, response.data.upload]); // add new upload to the list
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Upload a file</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button type="submit">Upload</button>
      </form>
      <h2>Uploads:</h2>
      <ul>
        {uploads.map((upload: { filename: string | number | boolean | ReactElement<any, string | JSXElementConstructor<any>> | Iterable<ReactNode> | ReactPortal | null | undefined; }, index: Key | null | undefined) => (
          <li key={index}>{upload.filename}</li>
        ))}
      </ul>
    </div>
  );
};

export default Upload;