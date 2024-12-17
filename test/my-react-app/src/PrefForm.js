import React from "react";
import { useState } from "react";

function PrefForm () {

  const [outfitType, setOutfitType] = useState(0);
  const [preference, setPreference] = useState(0);
  const [recommendation, setRecommendation] = useState(null);
  
  /*
  const getRecommendation = async () => {
    try {
        const response = await fetch("http://127.0.0.1:8000/recommend", {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
              "outfitType": "1",
              "preference": "1"
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
    
        // const data = await response.json();
        const data = { name: 'Red Skirt', popularity: 4.8 };
        setRecommendation(data);
    } catch (error) {
        console.error("Error fetching recommendation:", error);
    }
  };
  */

  
  const getRecommendation = async () => {
    const data = { name: 'Red Skirt', popularity: 4.8 };
    setRecommendation(data);
  }
    
  
    return (
      <div className="Form">
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
              <option value="1">Top</option>
              <option value="2">Pants</option>
              <option value="3">Dress</option>
              <option value="4">Accessories</option>
              <option value="5">Swimwear</option>
              <option value="6">Bags</option>
            </select>
          </label>
          <br />
          <label>
            How popular do you want your outfit to be?
            <select value={preference} onChange={(e) => setPreference(e.target.value)} required>
              <option value="">--Select--</option>
              <option value="0">Low</option>
              <option value="1">Medium</option>
              <option value="2">High</option>
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
  };

  export default PrefForm;