%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This cls file is adapted from LUMIA Lab Paper Style File (lumia.cls). We would like to express sincere thanks to the LUMIA team.
%
% This class file is designed for LUMIA Lab papers, technical reports, 
% and research publications. It provides a clean, professional layout with
% support for logos, author information, and links to code/model repositories.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lumia}[2025/03/01 LUMIA Lab Paper Style]

%%======== Package options and base class ========
\RequirePackage{xifthen}
\newboolean{nonumbering}
\newboolean{address}
\newboolean{copyright}
\newboolean{confidential}
\newboolean{uselogo}
\DeclareOption{nonumbering}{\setboolean{nonumbering}{true}}
\DeclareOption{address}{\setboolean{address}{true}}
\DeclareOption{copyright}{\setboolean{copyright}{true}}
\DeclareOption{confidential}{\setboolean{confidential}{true}}
\DeclareOption{logo}{\setboolean{uselogo}{true}}
\DeclareOption{nologo}{\setboolean{uselogo}{false}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ExecuteOptions{nologo} % Default: no logo
\ProcessOptions
\LoadClass{extarticle}

%%======== Packages Required ========
% Fonts and language
\AtEndOfClass{\RequirePackage[tracking=smallcaps]{microtype}}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage{amsthm, amsmath, amsfonts, amssymb}
\RequirePackage{cmap}
\RequirePackage[T1]{fontenc}
\RequirePackage{url}
\RequirePackage[colorlinks=true, allcolors=darkblue]{hyperref}
\RequirePackage{pifont}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{textcomp}
\RequirePackage[overload]{textcase}
% Graphics, tables and other formatting
\RequirePackage{graphicx}
\RequirePackage{colortbl}
\RequirePackage{booktabs}
% Define colors
\definecolor{darkblue}{rgb}{0, 0, 0.5}
\definecolor{lightblue}{rgb}{0.9, 0.95, 1.0}
\definecolor{lumiateal}{HTML}{005451}
\definecolor{lumiaabsbg}{HTML}{EDF6F6}

\definecolor{lumialink}{HTML}{239BA7}
% \hypersetup{colorlinks=true, allcolors=lumialink}
% \RequirePackage[colorlinks=true, allcolors=lumialink]{hyperref}
\hypersetup{colorlinks=true, allcolors=lumialink}

\RequirePackage{changepage}
\RequirePackage{enumitem}
\RequirePackage{tabularx}
\RequirePackage{datetime}
% Headers and footers
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
% Lists spacing
\setlist[itemize]{noitemsep, nosep}
\setlist[enumerate]{noitemsep}
% Title, Abstract, Bibliography
\RequirePackage[explicit]{titlesec}
\RequirePackage{bibentry}
\RequirePackage{caption}
\RequirePackage{needspace}
\RequirePackage{soul}
\RequirePackage{kvoptions}
% For colored background in abstract
\RequirePackage[skins, breakable, most]{tcolorbox}
% For icons (email, github, huggingface)
\RequirePackage{fontawesome5}

%%======== Font Setup ========
% Use Bitstream Charter as the main font
\RequirePackage{XCharter}
\RequirePackage[xcharter,bigdelims,vvarbb]{newtxmath}
\RequirePackage[scaled=1.1]{zlmtt}

%%======== Margins and column spacing ========
\RequirePackage[left=2.2cm,%
right=2.2cm,%
top=3cm,%
bottom=3cm,%
headheight=40pt,%
headsep=20pt,%
a4paper]{geometry}%
\setlength{\parindent}{0pt}
\setlength{\columnsep}{15pt}
\setlength{\parskip}{0.5\baselineskip}

%%======== Custom font sizes and styles ========
\newcommand{\watermarkfont}{\color{black50}\normalfont\fontsize{10}{12}\selectfont}
\newcommand{\headerfont}{\color{black}\normalfont\bfseries\fontsize{13}{14}\selectfont}
\newcommand{\titlefont}{\color{lumiateal}\normalfont\bfseries\fontsize{21}{23}\selectfont}
\newcommand{\datesfont}{\normalfont\sffamily\itshape\fontsize{8}{10}\selectfont}
\newcommand\urlheaderfont{\color{blue}\fontsize{8}{10}\selectfont}
\newcommand{\absfont}{\linespread{1.2}\fontsize{10}{12}\selectfont}
% \newcommand{\keywordstitlefont}{\bfseries\linespread{1.2}\fontsize{8}{10}\selectfont}
% \newcommand{\keywordsfont}{\itshape\linespread{1.2}\fontsize{11}{12}\selectfont}
\newcommand{\copyrightfont}{\normalfont\fontsize{7}{9}\selectfont}
\newcommand{\footerfont}{\color{black}\normalfont\fontsize{13}{14}\selectfont}
\newcommand{\subjectfont}{\color{white}\bfseries\fontsize{10}{12}\selectfont}
\newcommand{\headingfont}{\color{lumiateal}\bfseries\fontsize{13}{14}\selectfont}
% Font for contact info (email, github, huggingface)
\newcommand{\emailicon}{{\fontsize{8}{10}\selectfont\faEnvelope}}
\newcommand{\contactfont}{\normalfont\fontsize{8}{10}\selectfont}
\newcommand{\linkfont}{\normalfont\fontsize{8}{10}\selectfont}
\newcommand{\headertitle}{}
\newcommand{\setheadertitle}[1]{\renewcommand{\headertitle}{#1}}

%%======== Custom commands for metadata ========
\newtoks\correspondingauthor
\newtoks\correspondingemail
\newtoks\githublink
\newtoks\huggingfacelink
\newtoks\paperurl
\newtoks\reportnumber
\newtoks\headertext

% Command to set header text
\newcommand{\setheadertext}[1]{\headertext{#1}}

% Date format (ISO-8601)
\newdateformat{monthyeardate}{\THEYEAR-\twodigit{\THEMONTH}-\twodigit{\THEDAY}}

\setcounter{secnumdepth}{3}
\ifthenelse{\boolean{nonumbering}}{\setcounter{secnumdepth}{0}}{}

%%======== Author and affiliation ========
\RequirePackage{authblk}
\setcounter{Maxaffil}{0}
\setlength{\affilsep}{0.5em}
\renewcommand\Authfont{\normalfont\bfseries\fontsize{9}{13}\selectfont}
\renewcommand\Affilfont{\normalfont\fontsize{8}{10}\selectfont}
\makeatletter
\renewcommand\AB@affilsepx{, \protect\Affilfont}
\makeatother
\renewcommand\Authands{ and }

%%======== Commands for contact information ========
% Define HuggingFace icon command
\newcommand{\faHuggingface}{%
  \raisebox{-0.1em}{%
    \includegraphics[height=0.9em]{assets/hf.pdf}%
  }%
}

% Command to display corresponding author email(s)
\makeatletter
\newcommand{\lumia@correspondingemail}{%
  \ifdefined\correspondingemail
  \if\relax\the\correspondingemail\relax
  \else
    % \vskip2pt
    {\contactfont
    \the\correspondingemail\\
    }
  \fi
  \fi
}

% Command to display GitHub link
\newcommand{\lumia@githublink}{%
  \ifdefined\githublink
  \if\relax\the\githublink\relax
  \else
    % \vskip2pt
    {\linkfont
    \faGithub\ \href{\the\githublink}{\the\githublink}\quad\quad
    }
  \fi
  \fi
}

% Command to display HuggingFace link
\newcommand{\lumia@huggingfacelink}{%
  \ifdefined\huggingfacelink
  \if\relax\the\huggingfacelink\relax
  \else
    % \vskip2pt
    {\raggedright\linkfont
    \faHuggingface\ \href{\the\huggingfacelink}{\the\huggingfacelink}
    \par}
  \fi
  \fi
}
\makeatother

\newcommand{\internalonly}{\hl{Confidential---Do not share or forward externally}}
\newcommand{\copyrightext}{\footerfont \textcopyright\, \the\year{} LUMIA Lab. All rights reserved}

%%======== Header and Footer Content ========
\pagestyle{fancy}
\fancypagestyle{firststyle}{
    \fancyhead[L]{
        \IfFileExists{img/logo1.png}{%
            \includegraphics[height=26pt,keepaspectratio]{img/logo1.png}\hspace{10pt}%
            % \includegraphics[height=24pt,keepaspectratio]{img/logo2.png}\hspace{10pt}%
            \includegraphics[height=24pt,keepaspectratio]{img/logo3.png}%
        }{%
            \ifthenelse{\boolean{uselogo}}{
                \includegraphics[height=18pt]{assets/logo.png}\hspace{8pt}{\headerfont XXX}
            }{
                \ifdefined\headertext
                \if\relax\the\headertext\relax
                \else {\headerfont\the\headertext} \fi
                \else \fi
            }%
        }%
    }
    \fancyhead[R]{
        \ifdefined\paperurl
        \if\relax\the\paperurl\relax \else
            \href{\the\paperurl}{\urlheaderfont \itshape \the\paperurl}\\ \fi
        \else 
        \fi
        {\footerfont\itshape\monthyeardate\today}
        \ifthenelse{\boolean{confidential}}{\\ \footerfont \internalonly}{}
    }
    \fancyhead[C]{}
    %
    \fancyfoot[L]{
        \ifthenelse{\boolean{copyright}}{\copyrightext}{}
    }
    \fancyfoot[R]{
        \ifthenelse{\boolean{confidential}}{
        \ifdefined\reportnumber
        \if\relax\the\reportnumber\relax
        \else \footerfont\itshape  {\footerfont LUMIA Lab Technical Report \the\reportnumber} \fi
        \else \fi
        }{\footerfont\bfseries\relax}
    }
    \fancyfoot[C]{\footerfont\bfseries\relax}
}

% Header and Footer for pages from 2 onwards
\fancyhead[L,R]{}
\makeatletter
% \fancyhead[C]{\footerfont \@title}
% Default to use \title, but if \headertitle is set, use \headertitle instead
\fancyhead[C]{{\normalfont\fontsize{10}{11}\selectfont
  \ifdefempty{\headertitle}{\@title}{\headertitle}
}}
\makeatother
\fancyfoot[C]{}
\fancyfoot[L]{
    \ifthenelse{\boolean{confidential}}{\footerfont \internalonly}{\footerfont\bfseries\relax}
}
\fancyfoot[R]{\footerfont \thepage}

% Header and footer rules
\renewcommand{\headrulewidth}{1pt}
\renewcommand{\footrulewidth}{1pt}
\let\DefaultHeadRule\headrule
\let\DefaultFootRule\footrule
\renewcommand{\headrule}{\color{black}\DefaultHeadRule}
\renewcommand{\footrule}{\textcolor{black}{\DefaultFootRule}}

\renewenvironment{abstract}
{\par\noindent\\  \ignorespaces}
{\par\bigskip}

%%======== Bibliography ========
\addto{\captionsenglish}{\renewcommand{\refname}{References}}
\newcommand{\citeas}[1]{\Needspace*{8\baselineskip}\noindent\bibentry{#1}}

%%======== Abstract and section formatting ========
\def\xabstract{abstract}
\long\def\abstract#1\end#2{\def\two{#2}\ifx\two\xabstract
    \long\gdef\theabstract{\ignorespaces#1}
    \def\go{\end{abstract}}\else
\typeout{^^J^^J PLEASE DO NOT USE ANY \string\begin\space \string\end^^J
    COMMANDS WITHIN ABSTRACT^^J^^J}#1\end{#2}
\gdef\theabstract{\vskip12pt BADLY FORMED ABSTRACT: PLEASE DO
NOT USE {\tt\string\begin...\string\end} COMMANDS WITHIN
THE ABSTRACT\vskip12pt}\let\go\relax\fi
\go}

% Define abstract content with light blue background
\makeatletter
\newcommand{\abscontent}{
    \noindent
    \begin{tcolorbox}[
        colback=lumiaabsbg,
        colframe=lumiaabsbg,
        boxrule=0pt,
        arc=3pt,
        left=6pt,
        right=6pt,
        top=6pt,
        bottom=6pt
    ]
    \parbox{\dimexpr\linewidth-12pt}{\textbf{Abstract } \absfont \theabstract}
    \end{tcolorbox}
}

% Maketitle for two-column layout
\if@twocolumn%
\renewcommand{\maketitle}{
    \twocolumn[
    \begin{adjustwidth}{0pt}{24pt}
        \begin{flushleft}
            {
                {\raggedright \titlefont \@title\par}%
                \vskip11pt
                {\raggedright \@author\par}
                % \vskip2pt
                % Add corresponding email
                \lumia@correspondingemail
                % Add GitHub link
                \lumia@githublink
                % Add HuggingFace link
                \lumia@huggingfacelink
                \vskip4pt%
            }%
        \end{flushleft}
    \end{adjustwidth}
    % Add black line above abstract
    \noindent\textcolor{black}{\rule{\linewidth}{1pt}}
    \vskip8pt
        \abscontent
    \vskip20pt%
    ]
    \thispagestyle{firststyle}
}%
\else% Single column
\renewcommand{\maketitle}{
\vspace*{-35pt}%
\bgroup\setlength{\parindent}{0pt}
    \begin{adjustwidth}{0pt}{24pt}
        \begin{flushleft}
            {
                {\raggedright \titlefont \@title\par}%
                \vskip-10pt%
                \vskip11pt
                {\raggedright \@author\par}
                \vskip3pt%
                % Add corresponding email
                \lumia@correspondingemail
                % Add GitHub link
                \lumia@githublink
                % Add HuggingFace link
                \lumia@huggingfacelink
                \vskip8pt%
            }%
        \end{flushleft}
    \end{adjustwidth}
    \egroup
    % Add black line above abstract
    \vskip-20pt%
    \noindent\textcolor{black}{\rule{\linewidth}{1pt}}
    \vskip-4pt%
    \vskip8pt
    {%
        {\abscontent}
    }%
    \thispagestyle{firststyle}
}%
\fi
\makeatother

%%======== Section formatting ========
\titleformat{\section}
{\large\bfseries\headingfont}
{\thesection.}
{0.5em}
{#1}
[]
\titleformat{name=\section,numberless}
{\large\bfseries\headingfont}
{}
{0em}
{#1}
[]
\titleformat{\subsection}
{\color{lumiateal}\bfseries}
{\thesubsection.}
{0.5em}
{#1}
[]
\titleformat{\subsubsection}
{\color{lumiateal}\bfseries\itshape}
{\thesubsubsection.}
{0.5em}
{#1}
[]

\titlespacing*{\section}{0pc}{3ex plus4pt minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex plus3pt minus2pt}{2pt}
\titlespacing*{\subsubsection}{0pc}{2ex plus2.5pt minus1.5pt}{2pt}
\titleformat{\paragraph}[runin]{\bfseries}{}{-0.5em}{#1}
\titlespacing*{\paragraph}{0pt}{0ex}{.33em}

%%======== Captions ========
\DeclareCaptionLabelSeparator{pipe}{ $\vert$ }
\captionsetup{labelfont={bf},
labelsep=space,
figurename=Figure,
labelsep=pipe,
singlelinecheck=off,
}