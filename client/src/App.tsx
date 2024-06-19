import { BrowserRouter, Routes, Route} from 'react-router-dom';
import SigninForm from './pages/SigninForm';
import SignUpForm from './pages/SignupForm';
import HomePage from './pages/Home';
import Dashboard from './pages/Dashboard';

const App: React.FC = () => {

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage/>} />
        <Route path="/signin" element={<SigninForm />} />
        <Route path="/signup" element={<SignUpForm />} />
        <Route path="/dashboard" element={<Dashboard/>} />

      </Routes>
    </BrowserRouter>
  );
};

export default App;