import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const SignUpForm: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    phone: '',
    email: '',
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage(null);

    try {
      const response = await axios.post('http://localhost:5000/signup', formData);
      setMessage(response.data.message);
      navigate('/signin');
    } catch (error: any) {
      setMessage(error.response?.data?.error || 'An error occurred during sign up.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto m-4 p-6 bg-white rounded shadow-md">
      <h2 className="text-2xl font-bold mb-4">Sign Up</h2>
      {message && <p className={`mb-4 ${message.includes('error') ? 'text-red-500' : 'text-green-500'}`}>{message}</p>}
      <form onSubmit={handleSubmit}>
        {['username', 'password', 'phone', 'email'].map((field) => (
          <div key={field} className="mb-4">
            <label htmlFor={field} className="block text-gray-700 capitalize">{field}</label>
            <input
              type={field === 'password' ? 'password' : field === 'email' ? 'email' : 'text'}
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
          {isLoading ? 'Signing Up...' : 'Sign Up'}
        </button>
      </form>
    </div>
  );
};

export default SignUpForm;