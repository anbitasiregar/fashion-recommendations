import React from "react";
import { useState } from "react";

function PrefForm () {

  const [outfitType, setOutfitType] = useState(0);
  const [preference, setPreference] = useState(0);
  const [recommendation, setRecommendation] = useState(null);
  
  
  const getRecommendation = async () => {
    console.log("HERE: getrecommendation")
    try {
        const response = await fetch("http://localhost:8000/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Credentials": "true"
            },
            body: JSON.stringify({
              "outfitType": outfitType,
              "preference": preference
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
    
        const data = await response.json();
        setRecommendation(data);
    } catch (error) {
        console.error("Error fetching recommendation:", error);
    }
  };
  
    return (
      <div className="Form">
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