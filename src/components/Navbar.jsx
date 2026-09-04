import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Database, User, Brain } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import AuthButtons from "./auth/AuthButtons";
import DocsLink from "./ui/DocsLink";
import ReportBugLink from "./ui/ReportBugLink";
import ThemeToggle from "./ui/ThemeToggle";
import Wordmark from './Wordmark';

// Nav items are uniform, so the active/idle treatment lives in one place rather
// than being repeated per button.
const navItemClass = (active) =>
  [
    "flex items-center gap-[7px] px-[11px] h-[30px] rounded-6 text-btn font-medium",
    "transition-colors duration-150",
    active ? "text-ac bg-acS" : "text-t2 hover:text-t1 hover:bg-hv",
  ].join(" ");

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, user } = useAuth();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-glass backdrop-blur-md border-b border-ln">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div
            className="flex items-center gap-[8px] cursor-pointer"
            onClick={() => navigate('/')}
          >
            <div className="w-8 h-8 bg-accent rounded-7 flex items-center justify-center">
              <Database className="w-[18px] h-[18px] text-onAccent" />
            </div>
            <span
              className="text-xl font-semibold tracking-tight text-t1 cursor-pointer transition-colors duration-150 hover:text-ac"
              onClick={(e) => {
                e.stopPropagation();
                navigate('/');
              }}
            >
              <Wordmark />
            </span>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-[4px]">
            {/* Username Display */}
            {isAuthenticated && user && (
              <div className="flex items-center gap-[6px] px-[10px] text-btn text-t3">
                <User className="w-[14px] h-[14px]" />
                <span className="font-medium">{user.username}</span>
              </div>
            )}

            {isAuthenticated && (
              <button
                onClick={() => navigate('/datasets')}
                className={navItemClass(isActive('/datasets'))}
              >
                <Database className="w-[14px] h-[14px]" />
                <span>Datasets</span>
              </button>
            )}

            <button
              onClick={() => navigate('/models')}
              className={navItemClass(isActive('/models'))}
            >
              <Brain className="w-[14px] h-[14px]" />
              <span>Models</span>
            </button>

            <DocsLink className={navItemClass(false)} />

            <ReportBugLink
              className={navItemClass(false)}
              textColor="text-t2"
              bgColor="hover:bg-hv"
            />

            <ThemeToggle />

            {/* Auth Section */}
            <div className="flex items-center gap-[8px] ml-[8px] pl-[12px] border-l border-ln">
              <AuthButtons
                textColor="text-t2"
                buttonClass={isAuthenticated
                  ? "flex items-center gap-[7px] px-[11px] h-[30px] rounded-6 text-btn font-medium text-t2 hover:text-err hover:bg-errBg transition-colors duration-150"
                  : "flex items-center gap-[7px] px-[11px] h-[30px] rounded-6 text-btn font-medium text-onAccent bg-accent hover:brightness-110 transition-all duration-150"
                }
                showLogoutOnly={true}
              />
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
