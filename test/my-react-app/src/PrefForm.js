import React from "react";
import { useState } from "react";

function PrefForm () {

  const [outfitType, setOutfitType] = useState(0);
  const [preference, setPreference] = useState(0);
  
    return (
      <div className="Form">
        <h1>Outfit Recommender</h1>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            // getRecommendation();
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
      </div>
    );
  };

  export default PrefForm;