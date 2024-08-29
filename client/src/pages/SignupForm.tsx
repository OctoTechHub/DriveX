import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Navbar from './components/Navbar';

const SignUpForm: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    phone: '',
    email: '',
  });
  const [message, setMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage(null);

    try {
      const response = await axios.post<{ message: string }>('http://localhost:5000/signup', formData);
      setMessage(response.data.message);
      navigate('/signin');
    } catch (error: unknown) {
      if (axios.isAxiosError(error) && error.response) {
        setMessage(error.response.data.error || 'An error occurred during sign up.');
      } else {
        setMessage('An error occurred during sign up.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
    <Navbar/>
    <div className="relative overflow-hidden font-mono h-screen flex items-center justify-center">
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

      {/* Sign Up Form */}
      <div className="relative bg-white p-6 rounded-lg shadow-md max-w-md w-full">
        <h2 className="text-3xl font-bold mb-6 text-center">Sign Up</h2>
        {message && (
          <p
            className={`mb-4 text-center ${message.includes('error') ? 'text-red-500' : 'text-green-500'}`}
            aria-live="polite"
          >
            {message}
          </p>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          {['username', 'password', 'phone', 'email'].map((field) => (
            <div key={field} className="mb-4">
              <label htmlFor={field} className="block text-gray-700 font-medium capitalize">
                {field}
              </label>
              <input
                type={field === 'password' ? 'password' : 'text'}
                id={field}
                name={field}
                value={formData[field as keyof typeof formData]}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
                aria-label={field}
              />
            </div>
          ))}
          <button
            type="submit"
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition duration-150 ease-in-out"
            disabled={isLoading}
            aria-label="Sign Up"
          >
            {isLoading ? 'Signing Up...' : 'Sign Up'}
          </button>
        </form>
      </div>
    </div>
    </>
  );
};

export default SignUpForm;
