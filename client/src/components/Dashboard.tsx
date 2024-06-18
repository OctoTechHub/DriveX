import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Dashboard: React.FC = () => {
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

  return (
    <div>
      <h1>Welcome to your dashboard, {localStorage.getItem('username')}!</h1>
      <ul>
        {uploads.map((upload: any) => (
          <li key={upload.filename}>
            {upload.filename} - {upload.upload_date}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Dashboard;