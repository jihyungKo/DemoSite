import './App.css';

import { Route, Redirect, Switch } from 'react-router-dom';
import { ConnectedRouter } from 'connected-react-router';
import { connect } from 'react-redux';

import Main from './containers/Main/Main'; 
import History from './containers/History/History/History';
import HistoryDetail from './containers/History/HistoryDetail/HistoryDetail';

function App(props) {
  const { history } = props;
  
  return (
    <div className="App">
      <ConnectedRouter history = {history}>
        <Switch>
          <Route path="/main" exact component={Main}/>
          <Route path="/history" exact component={History}/>
          <Route path="/history/:id" exact component={HistoryDetail}/>
          <Redirect from="/" to ="/main" />
        </Switch>
      </ConnectedRouter>
    </div>
  );
}

export default connect(
  mapStateToProps,
  null,
)(App);
