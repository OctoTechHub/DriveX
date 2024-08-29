import React from 'react';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {
  return (
    <div className="relative overflow-hidden w-full h-screen font-mono">
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

      {/* Navbar with Logo */}
      <nav className="absolute top-0 left-0 right-0 flex justify-between items-center p-4 bg-transparent">
        <img src="/assets/logo.png" alt="Logo" className="h-10" /> {/* Update the path */}
        <div className="space-x-4">
          <Link to="/" className="text-white hover:text-gray-300">
            Home
          </Link>
          <Link to="/about" className="text-white hover:text-gray-300">
            About
          </Link>
          <Link to="/contact" className="text-white hover:text-gray-300">
            Contact
          </Link>
        </div>
      </nav>

      <div className="relative flex flex-col items-center justify-center w-full h-full p-6 backdrop-blur-md bg-opacity-70">
        <div className="text-center space-y-6">
          <h1 className="text-4xl font-bold text-gray-900">Welcome to DriveX</h1>
          <p className="text-lg text-gray-800 max-w-md mx-auto">
            Your all-in-one solution for secure and smart file management. Manage, share, and organize your files with ease.
          </p>
        </div>

        {/* AI-Based File System Glass Box */}


        <div className="flex flex-col items-center space-y-4 mt-10">
          <Link
            to="/signup"
            className="bg-blue-700 hover:bg-blue-500 text-white font-bold py-2 px-6 rounded shadow"
          >
            New User? Sign Up
          </Link>
          <Link
            to="/signin"
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded shadow"
          >
            Existing User? Sign In
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="absolute bottom-0 left-0 right-0 p-4 bg-transparent text-center text-white">
        <p className="text-sm">
          &copy; {new Date().getFullYear()} DriveX. All rights reserved.
        </p>
      </footer>
    </div>
  );
};

export default HomePage;
