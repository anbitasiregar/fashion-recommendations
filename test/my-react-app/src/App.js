import logo from './logo.svg';
import './App.css';
import PrefForm from './PrefForm';

function App() {
  return (
    <div className="App">
        <h1 className="Heading">Outfit Recommender</h1>
        <iframe src="https://giphy.com/embed/3o6wrsNuUBkjdzsntK" width="480" height="271" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/filmeditor-clueless-movie-3o6wrsNuUBkjdzsntK">via GIPHY</a></p>
        <PrefForm className="Form" />
      </div>
  );
}

export default App;
