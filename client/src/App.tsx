import { BrowserRouter, Routes, Route} from 'react-router-dom';
import SigninForm from './components/SigninForm';
import SignUpForm from './components/SignupForm';
import HomePage from './components/Home';
import Upload from './components/Upload';
import Dashboard from './components/Dashboard';

const App: React.FC = () => {

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage/>} />
        <Route path="/signin" element={<SigninForm />} />
        <Route path="/signup" element={<SignUpForm />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/dashboard" element={<Dashboard/>} />

      </Routes>
    </BrowserRouter>
  );
};

export default App;