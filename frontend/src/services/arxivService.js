import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const fetchLatestUpdates = async () => {
  try {
    // Now we'll use the actual API endpoint
    const response = await axios.get(`${API_URL}/updates`);
    return response.data;
  } catch (error) {
    console.error('Error fetching arXiv updates:', error);
    throw error;
  }
};