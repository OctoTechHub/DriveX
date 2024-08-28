import React, { useState, FormEvent } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

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
      const response = await axios.post('http://localhost:5000/login', formData, { withCredentials: true });
      setMessage(response.data.message);
      
      // Store only the username in localStorage, not the password
      localStorage.setItem('username', formData.username);
      
      navigate('/dashboard', { replace: true });
    } catch (error: any) {
      setMessage(error.response?.data?.error || 'An error occurred during sign in.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto m-4 p-6 bg-white rounded shadow-md">
      <h2 className="text-2xl font-bold mb-4">Sign In</h2>
      {message && <p className={`mb-4 ${message.includes('error') ? 'text-red-500' : 'text-green-500'}`}>{message}</p>}
      <form onSubmit={handleSubmit}>
        {['username', 'password'].map((field) => (
          <div key={field} className="mb-4">
            <label htmlFor={field} className="block text-gray-700 capitalize">{field}</label>
            <input
              type={field === 'password' ? 'password' : 'text'}
              id={field}
              name={field}
              className="form-input mt-1 block w-full"
              required
              value={formData[field as keyof typeof formData]}
              onChange={handleChange}
            />
          </div>
        ))}
        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          disabled={isLoading}
        >
          {isLoading ? 'Signing In...' : 'Sign In'}
        </button>
      </form>
    </div>
  );
};

export default SignInForm;