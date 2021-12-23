import React, { Component } from 'react';
import { connect } from 'react-redux';

import * as actionCreators from '../../store/actions/index.js/index';

import './Main.css'

class Main extends Component {
    constructor(props){
        super(props);
    }

    async componentDidMount(){}

    render(){
        const {} = this.props;
        return(
            <div
                className="Main"
            />
        )
    }
}

export default connect(
    mapStateToProps,
    mapDispatchToProps,
)(Main);