import React, { createContext, useState } from "react";

// Create a new context
export const StateContext = createContext();

// Create a provider component
export const StateProvider = ({ children }) => {
  // Define your state variables here
  const [selectedTab, setSelectedTab] = useState("Home");

  // Define any functions or methods you need for state management

  // Return the provider component with the state and any functions/methods you want to expose
  return (
    <StateContext.Provider value={{ selectedTab, setSelectedTab }}>
      {children}
    </StateContext.Provider>
  );
};
