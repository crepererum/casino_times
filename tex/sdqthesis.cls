%% LaTeX2e class for student theses
%% sdthesis.cls
%%
%% Karlsruhe Institute of Technology
%% Institute for Program Structures and Data Organization
%% Chair for Software Design and Quality (SDQ)
%%
%% Dr.-Ing. Erik Burger
%% burger@kit.edu
%%
%% Version 1.1, 2014-11-21
%%
%% SVN-Informationen:
%% ------------------
%% $LastChangedBy: burger $
%% $LastChangedRevision: 3448 $
%% $LastChangedDate: 2014-11-21 10:55:23 +0100 (Fri, 21 Nov 2014) $
%% $HeadURL: https://svnserver.informatik.kit.edu/i43/svn/lehre/Abschlussarbeiten-Vorlage/sdqthesis.cls $

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdqthesis}[2014/10/21 v1.0 Student thesis class]
\DeclareOption{ngerman}{\PassOptionsToPackage{\CurrentOption}{babel}}
% German is also needed in English thesis for the abstract
\DeclareOption{english}{\PassOptionsToPackage{main=\CurrentOption,ngerman}{babel}}
\ProcessOptions\relax

%% ---------------------------------------------------------------
%% | Based on KOMAscript by Markus Kohm http://www.komascript.de |
%% ---------------------------------------------------------------
\LoadClass[12pt,twoside,headsepline,a4paper,titlepage=true,DIV=15,BCOR=20mm]{scrbook}

%% -----------------------
%% | Headers and footers |
%% -----------------------
\RequirePackage[automark]{scrpage2}

\clubpenalty=10000 %prevent orphans
\widowpenalty=10000 %prvent widows

%% --------------
%% | Typography |
%% --------------

\RequirePackage{fontspec}
\defaultfontfeatures{
    Ligatures=TeX
}
%\defaultfontfeatures{Ligatures=Rare}
\setmainfont{LinLibertine_R}[
    Path           = /usr/share/fonts/OTF/,
    Extension      = .otf,
    BoldFont       = LinLibertine_RB,
    ItalicFont     = LinLibertine_RI,
    BoldItalicFont = LinLibertine_RBI
]
\setsansfont{SourceSansPro-Regular}[
    Path           = /usr/share/fonts/OTF/,
    Extension      = .otf,
    BoldFont       = SourceSansPro-Bold,
    ItalicFont     = SourceSansPro-It,
    BoldItalicFont = SourceSansPro-BoldIt
]
\setmonofont{VeraMono}[
    Path           = /usr/share/fonts/TTF/,
    Extension      = .ttf,
    BoldFont       = VeraMoBd,
    ItalicFont     = VeraMoIt,
    BoldItalicFont = VeraMoBI,
    Scale          = 0.78
]

% serif type: Linux Libertine
%\RequirePackage{libertine}
% Linux Libertine in math mode
%\RequirePackage[libertine]{newtxmath}

% grotesque type: Adobe Source Sans Pro
%\RequirePackage[scaled=.92]{sourcesanspro}

% monospace type: Bera Mono
%\RequirePackage[scaled=.78]{beramono}

% semi-bold type (for subsections and paragraphs)
\newcommand*{\sbdefault}{sb}
\DeclareRobustCommand{\sbseries}{%
  \not@math@alphabet\sbseries\relax
  \fontseries\sbdefault\selectfont}

\DeclareTextFontCommand{\textsb}{\sbseries}

\addtokomafont{subsection}{\sbseries}
\addtokomafont{subsubsection}{\sbseries}
\addtokomafont{paragraph}{\sbseries}

% microtype for nicer typography
\RequirePackage[activate={true,nocompatibility},final]{microtype}

% commands for code and model keywords in text
\newcommand{\code}[1]{\texttt{\hyphenchar\font45\relax #1}}
\newcommand{\model}[1]{\textsf{#1}}

% nicer spacing for enumerations
\RequirePackage{enumitem}


%% --------------
%% | Title page |
%% --------------

\RequirePackage{tikz}
\usetikzlibrary{
    arrows,
    calc,
    fit,
    positioning
}
\newcommand{\xdiameter}{15}
\newcommand{\xone}{-15}
\newcommand{\xtwo}{145}
\newcommand{\yone}{25}
\newcommand{\ytwo}{-243}

% Logos
\newcommand{\thegrouplogo}{sdqlogo}
\newcommand{\grouplogo}[1]{\renewcommand{\thegrouplogo}{#1}}
\newcommand{\printgrouplogo}{\includegraphics[height=1.8cm]{logos/\thegrouplogo}}
\newcommand{\nogrouplogo}{\renewcommand{\printgrouplogo}{}}

\titlehead{%
\begin{tikzpicture}[overlay]
\draw[color=gray]
 		 (\xone mm, \yone mm)
  -- (\xtwo mm, \yone mm)
 arc (90:0:\xdiameter mm)
  -- (\xtwo mm + \xdiameter mm , \ytwo mm)
	-- (\xone mm + \xdiameter mm , \ytwo mm)
 arc (270:180:\xdiameter mm)
	-- (\xone mm, \yone mm);
\end{tikzpicture}
% KIT logo german/english
\begin{center}
\iflanguage{ngerman}
{\includegraphics[height=2.8cm]{logos/kitlogo_de_cmyk}}
{\includegraphics[height=2.8cm]{logos/kitlogo_en_cmyk}}
\end{center}
%\hfill
% SDQ logo
%\printgrouplogo
%\ThisCenterWallPaper{1}{title-background.pdf}
}

%
\setkomafont{title}{\huge\sffamily\bfseries}
\setkomafont{subtitle}{\normalfont\large}

\uppertitleback{Karlsruher Institut für Technologie\\ Fakultät für Informatik\\ Postfach 6980\\ 76128 Karlsruhe}

%% variables for title page
\newcommand{\theinstitute}{\iflanguage{english}
    {Institute for Program Structures and Data Organization (IPD)}
    {Institut für Programmstrukturen und Datenorganisation (IPD)}}
\newcommand{\thethesistype}{}
\newcommand{\thereviewerone}{}
\newcommand{\thereviewertwo}{}
\newcommand{\theadvisorone}{}
\newcommand{\theadvisortwo}{}
\newcommand{\thesubmissiondate}{}
\newcommand{\thepresentationdate}{}

%% formatting commands for titlepage
\newcommand{\thesistype}[1]{\subtitle{\vskip2em #1 \iflanguage{english}{of}{von}}%
\renewcommand{\thethesistype}{#1}}
\newcommand{\myinstitute}[1]{\renewcommand{\theinstitute}{#1}}
\newcommand{\reviewerone}[1]{\renewcommand{\thereviewerone}{#1}}
\newcommand{\reviewertwo}[1]{\renewcommand{\thereviewertwo}{#1}}
\newcommand{\advisorone}[1]{\renewcommand{\theadvisorone}{#1}}
\newcommand{\advisortwo}[1]{\renewcommand{\theadvisortwo}{#1}}
\newcommand{\submissiondate}[1]{\renewcommand{\thesubmissiondate}{#1}\date{}}
\newcommand{\presentationdate}[1]{\renewcommand{\thepresentationdate}{#1}\date{}}

\newcommand{\settitle}{%
\publishers{%
\large
\iflanguage{english}{at the Department of Informatics}%
{an der Fakultät für Informatik}\\
\theinstitute\\[2em]
\begin{tabular}{l l}
    \iflanguage{english}{Reviewer}{Erstgutachter}: & \thereviewerone\\
    \iflanguage{english}{Second reviewer}{Zweitgutachter}: & \thereviewertwo\\
    \iflanguage{english}{Advisor}{Betreuender Mitarbeiter}: &  \theadvisorone\\
    % if there is no second advisor, do not output this line
    \ifthenelse{\equal{\theadvisortwo}{}}{}{%
    \iflanguage{english}{Second advisor}{Zweiter betreuender Mitarbeiter}: & \theadvisortwo\\
    }
    \noalign{\vskip2em}
    Submission Date: & \thesubmissiondate\\
    Presentation Date: & \thepresentationdate\\
\end{tabular}
}
}


%% -----------------------------
%% | Abstract/Acknowledgements |
%% -----------------------------

\newcommand{\abstract}[1][\abstractname]{\chapter*{#1}}
\newcommand{\Abstract}[1][\abstractname]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}}

\def\ackname{Acknowledgments}
\def\abstractname{Abstract}
\def\switcht@deutsch{\svlanginfo
	\def\ackname{Danksagung}
	\def\abstractname{Kurzfassung}
}
\def\switcht@english{\svlanginfo
	\def\ackname{Acknowledgements}
	\def\abstractname{Abstract}
}

%% In English theses, an additional German Abstract is needed.
\newcommand{\includeabstract}{
\iflanguage{english}{
% include English and German abstracts
\include{sections/abstract_en}
\begin{otherlanguage}{ngerman}
\include{sections/abstract_de}
\end{otherlanguage}
}{
% include only German abstract
\include{sections/abstract_de}
}
}

%% ------------
%% | Packages |
%% ------------

% draft mode
\RequirePackage{ifdraft}
\RequirePackage{ifthen}

% enumerate subsubsections
\setcounter{secnumdepth}{3}

% subcaption
\RequirePackage{subcaption}

% wallpaper for title page (KIT-Frame)
\RequirePackage{wallpaper}

% subimport of files
\RequirePackage{import}

% languages
\RequirePackage{babel}

% hyphenation for compound words
\RequirePackage[shortcuts]{extdash}

% nice tables
\RequirePackage{booktabs}
\RequirePackage{longtable}
\RequirePackage{array}

% show graphics in draft mode
\RequirePackage{graphicx}
\setkeys{Gin}{draft=false}

% % Mathematics package
\RequirePackage{amsmath}
\let\openbox\relax
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{mathtools}
\RequirePackage{siunitx}
\RequirePackage{unicode-math}
\setmathfont[Scale=0.90]{Asana-Math.otf}

% easier quotes
\RequirePackage{csquotes}

% graphcis
\definecolor{kit-green}{cmyk}{0.1,0,0.6,0.2}
\definecolor{kit-blue}{cmyk}{0.8,0.5,0,0.2}
\definecolor{kit-green2}{cmyk}{0.6,0,1,0.2}
\definecolor{kit-yellow}{cmyk}{0,0.05,1,0.2}
\definecolor{kit-orange}{cmyk}{0,0.45,1,0.2}
\definecolor{kit-brown}{cmyk}{0.35,0.5,1,0.2}
\definecolor{kit-red}{cmyk}{0.25,1,1,0.2}
\definecolor{kit-violet}{cmyk}{0.25,1,0,0.2}
\definecolor{kit-cyan}{cmyk}{0.9,0.05,0,0.2}
\RequirePackage{pgfplots}
\pgfplotsset{
    compat=1.13,
    lua backend=true,
    discard if not/.style 2 args={
        x filter/.code={
            \edef\tempa{\thisrow{#1}}
            \edef\tempb{#2}
            \ifx\tempa\tempb
            \else
                \def\pgfmathresult{inf}
            \fi
        }
    },
    x_year/.style={
        x tick label style={
            /pgf/number format/.cd,
            fixed,
            set thousands separator={},
            /tikz/.cd
        }
    },
    fullwidth/.style={
        height=0.45\textwidth,
        width=\textwidth
    },
    filter discard warning=false,
    every tick label/.append style={font=\scriptsize},
    legend style={font=\scriptsize},
    label style={font=\scriptsize}
}

% appendix
\RequirePackage[toc,title,header]{appendix}
\noappendicestocpagenum

% todo-notes
\RequirePackage[obeyDraft]{todonotes}
\presetkeys{todonotes}{inline}{}

% PDF specific packages
\RequirePackage[hyphens]{url}
\RequirePackage[breaklinks,colorlinks=false,linktoc=all,final]{hyperref}
\newcommand\setpdf{\hypersetup{%
    pdftitle={\thethesistype},%
    pdfsubject={\@title},%
    pdfauthor={\@author},%
    pdfborder={0 0 0},%
    %linkcolor=kit-blue100,%
    %citecolor=kit-green100,%
    %urlcolor=kit-red100
    }%
    \let\theauthor\@author
}

% better tables
\RequirePackage{tabularx}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}

% algorithm
\RequirePackage[
    ruled,% nice lines above/beyond the algorithms
    linesnumbered% draw line numbers
]{algorithm2e}

%% automatic naming of sections with autoref
\def\sectionautorefname{Section}
\def\subsectionautorefname{Subsection}
\def\subsubsectionautorefname{Subsection}
\def\paragraphautorefname{Paragraph}
