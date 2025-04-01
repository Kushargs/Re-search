import React, { useEffect, useState } from 'react';
import { fetchLatestUpdates } from '../services/arxivService';

const UpdatesList = () => {
  const [updates, setUpdates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const getUpdates = async () => {
      try {
        const data = await fetchLatestUpdates();
        setUpdates(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch updates. Please try again later.');
        setLoading(false);
      }
    };

    getUpdates();
  }, []);

  if (loading) return <div>Loading latest arXiv updates...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="updates-container">
      <h2>Latest arXiv Updates</h2>
      <div className="updates-list">
        {updates.map((paper) => (
          <div key={paper.id} className="paper-card">
            <h3>{paper.title}</h3>
            <p className="authors">
              <strong>Authors:</strong> {paper.authors.join(', ')}
            </p>
            <p className="summary">{paper.summary}</p>
            <p className="date">
              <strong>Updated:</strong> {paper.updated}
            </p>
            <a 
              href={`https://arxiv.org/abs/${paper.id}`} 
              target="_blank" 
              rel="noopener noreferrer"
              className="view-button"
            >
              View on arXiv
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UpdatesList;