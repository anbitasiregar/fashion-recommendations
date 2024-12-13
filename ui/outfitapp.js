// Import dependencies
import { useState } from 'react';
import App from './App';

function App() {
  const [outfitType, setOutfitType] = useState('');
  const [preference, setPreference] = useState('popular');
  const [recommendation, setRecommendation] = useState(null);

  // Simulate recommendation function (replace with backend call later)
  const getRecommendation = async () => {
    try {
      const response = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          outfitType,
          preference,
        }),
      });
  
      const data = await response.json();
      setRecommendation(data);
    } catch (error) {
      console.error("Error fetching recommendation:", error);
    }
  };

  return (
    <div className="App">
      <h1>Outfit Recommender</h1>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          getRecommendation();
        }}
      >
        <label>
          What type of outfit are you looking for?
          <select value={outfitType} onChange={(e) => setOutfitType(e.target.value)} required>
            <option value="">--Select--</option>
            <option value="Skirt">Skirt</option>
            <option value="Pants">Pants</option>
            <option value="Dress">Dress</option>
          </select>
        </label>
        <br />
        <label>
          Do you want something popular or alternative?
          <select value={preference} onChange={(e) => setPreference(e.target.value)} required>
            <option value="popular">Popular</option>
            <option value="alternative">Alternative</option>
          </select>
        </label>
        <br />
        <button type="submit">Get Recommendation</button>
      </form>

      {recommendation && (
        <div className="recommendation">
          <h2>Recommended Outfit</h2>
          <p><strong>Name:</strong> {recommendation.name}</p>
          <p><strong>Popularity Score:</strong> {recommendation.popularity}</p>
        </div>
      )}
    </div>
  );
}

export default App;