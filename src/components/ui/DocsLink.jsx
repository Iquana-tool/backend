import React from 'react';
import { BookOpen } from 'lucide-react';
import { DOCS, docsUrl } from '../../constants/docs';

/**
 * The "Documentation" entry point, shared by every header that carries one.
 *
 * A real anchor rather than a `navigate()` button: the docs are a separate site,
 * so middle-click, "open in new tab" and "copy link address" should all behave
 * the way they do for any other outbound link. Deliberately the same shape as
 * `ReportBugLink`, which sits next to it in all four headers.
 *
 * @param {Object} props
 * @param {string} props.to - Docs path to open, from `DOCS`. Defaults to the home page.
 * @param {string} props.label - Link text.
 * @param {string} props.variant - 'default' | 'mobile' - Controls sizing of the default style.
 * @param {string} props.className - Custom className override
 * @param {string} props.textColor - Custom text color class
 * @param {string} props.bgColor - Custom background color class
 * @param {boolean} props.hideTextOnMobile - If true, hides text on mobile screens
 */
const DocsLink = ({
  to = DOCS.home,
  label = 'Documentation',
  variant = 'default',
  className,
  textColor = 'text-t2',
  bgColor = 'bg-hv hover:bg-hv2',
  hideTextOnMobile = false,
}) => {
  const isMobile = variant === 'mobile';
  const iconSize = isMobile ? 'w-3 h-3' : 'w-4 h-4';
  const textSize = isMobile ? 'text-xs' : 'text-sm';
  const spaceClass = isMobile ? 'space-x-1' : 'space-x-2';
  const padding = isMobile ? 'py-1.5 px-2' : 'py-2 px-4';

  const defaultClassName = `flex items-center ${spaceClass} ${bgColor} ${textColor} ${padding} rounded-6 transition-colors duration-150 ${textSize}`;

  const finalClassName = className || defaultClassName;
  const textDisplayClass = hideTextOnMobile ? 'hidden md:inline' : '';

  return (
    <a
      href={docsUrl(to)}
      target="_blank"
      rel="noopener noreferrer"
      className={finalClassName}
    >
      <BookOpen className={iconSize} />
      <span className={textDisplayClass}>
        {isMobile ? 'Docs' : label}
      </span>
    </a>
  );
};

export default DocsLink;
