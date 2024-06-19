import React from 'react';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {
  return (
    <div className="max-w-md mx-auto m-4 p-6 bg-white rounded shadow-md">
      <nav className="flex justify-between mb-4">
        <Link to="/" className="text-blue-500 hover:text-blue-700">
          Home
        </Link>
        <Link to="/signup" className="text-blue-500 hover:text-blue-700">
          Sign Up
        </Link>
        <Link to="/signin" className="text-blue-500 hover:text-blue-700">
          Sign In
        </Link>
      </nav>
      <h1 className="text-3xl font-bold mb-4">Welcome to DriveAI !</h1>
      <p className="text-gray-700 mb-4">This is the home page.</p>
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        onClick={() => console.log('Button clicked!')}
      >
        Click me!
      </button>
    </div>
  );
};

export default HomePage;