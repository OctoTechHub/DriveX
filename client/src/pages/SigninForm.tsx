import React, { useState, FormEvent } from 'react';
import axios, { AxiosError } from 'axios';
import { useNavigate } from 'react-router-dom';
import Navbar from './components/Navbar';

const SignInForm: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });
  const [message, setMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage(null);

    try {
      const response = await axios.post<{ message: string }>('http://localhost:5000/login', formData, { withCredentials: true });
      setMessage(response.data.message);
      
      localStorage.setItem('username', formData.username);
      
      navigate('/dashboard', { replace: true });
    } catch (error) {
      if (error instanceof AxiosError && error.response) {
        setMessage(error.response.data.error || 'An error occurred during sign in.');
      } else {
        setMessage('An unexpected error occurred.');
      }
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <>
    <Navbar/>
    <div className="relative overflow-hidden h-screen flex font-mono items-center justify-center">
      {/* Background Video */}
      <video
        autoPlay
        loop
        muted
        className="absolute inset-0 w-full h-full object-cover filter blur-sm"
      >
        <source src="/src/assets/video.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {/* Sign In Form */}
      <div className="relative bg-white p-6 rounded-lg shadow-md max-w-md w-full">
        <h2 className="text-3xl font-bold mb-6 text-center">Sign In</h2>
        {message && (
          <p
            className={`mb-4 text-center ${message.includes('error') ? 'text-red-500' : 'text-green-500'}`}
            aria-live="polite"
          >
            {message}
          </p>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          {['username', 'password'].map((field) => (
            <div key={field} className="mb-4">
              <label htmlFor={field} className="block text-gray-700 font-medium capitalize">
                {field}
              </label>
              <input
                type={field === 'password' ? 'password' : 'text'}
                id={field}
                name={field}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
                value={formData[field as keyof typeof formData]}
                onChange={handleChange}
                aria-label={field}
              />
            </div>
          ))}
          <button
            type="submit"
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition duration-150 ease-in-out"
            disabled={isLoading}
            aria-label="Sign In"
          >
            {isLoading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
    </>
  );
};

export default SignInForm;
