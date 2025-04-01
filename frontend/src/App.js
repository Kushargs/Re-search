import React from 'react';
import './App.css';
import UpdatesList from './components/UpdatesList';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>arXiv Updates</h1>
      </header>
      <main>
        <UpdatesList />
      </main>
      <footer>
        <p>Data sourced from arXiv.org</p>
      </footer>
    </div>
  );
}

export default App;